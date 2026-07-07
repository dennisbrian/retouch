"""Tests for retouch/image_analyzer.py — ImageAnalyzer (F9)."""

from __future__ import annotations

import numpy as np
import pytest

from retouch.image_analyzer import (
    ImageAnalysis,
    ImageAnalyzer,
    SkinCondition,
    LIGHTING_TYPES,
)
from retouch.params import param_names
from retouch.recipes import RECIPES


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------


def _gradient(h: int = 256, w: int = 256) -> np.ndarray:
    """Neutral grayscale gradient (no WB cast, low noise, normal key)."""
    yy = np.linspace(0, 255, w, dtype=np.float32)
    xx = np.linspace(0, 255, h, dtype=np.float32)
    grid = np.minimum(xx[:, None], yy[None, :])
    img = np.stack([grid, grid, grid], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _bright_flat(h: int = 256, w: int = 256, value: int = 220) -> np.ndarray:
    """Bright flat field — high-key studio look."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    return img


def _dark_noisy(h: int = 256, w: int = 256) -> np.ndarray:
    """Dark, noisy image — low-light scene."""
    rng = np.random.default_rng(42)
    base = np.full((h, w, 3), 40, dtype=np.float32)
    noise = rng.normal(0.0, 18.0, size=(h, w, 3)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img


def _warm_cast(h: int = 256, w: int = 256) -> np.ndarray:
    """Image with a warm WB cast (high R, low B)."""
    yy = np.linspace(0, 255, w, dtype=np.float32)
    xx = np.linspace(0, 255, h, dtype=np.float32)
    grid = np.minimum(xx[:, None], yy[None, :])
    b = grid * 0.7
    g = grid * 0.9
    r = grid * 1.0
    img = np.stack([b, g, r], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _cool_cast(h: int = 256, w: int = 256) -> np.ndarray:
    """Image with a cool WB cast (high B, low R)."""
    yy = np.linspace(0, 255, w, dtype=np.float32)
    xx = np.linspace(0, 255, h, dtype=np.float32)
    grid = np.minimum(xx[:, None], yy[None, :])
    b = grid * 1.0
    g = grid * 0.9
    r = grid * 0.7
    img = np.stack([b, g, r], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _skin_mask_block(h: int = 256, w: int = 256) -> np.ndarray:
    """A simple rectangular skin mask (float32 [0,1])."""
    mask = np.zeros((h, w), dtype=np.float32)
    mask[60:200, 80:180] = 1.0
    return mask


def _skin_tone_block(h: int = 256, w: int = 256) -> np.ndarray:
    """Image with a skin-tone block on a neutral background."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    # Skin tone in BGR ≈ (140, 170, 200) — warmish, in the expected skin locus
    img[60:200, 80:180] = (140, 170, 200)
    return img


# ---------------------------------------------------------------------------
# Tests: analyze produces valid ImageAnalysis
# ---------------------------------------------------------------------------


class TestAnalyzeProducesValidImageAnalysis:
    def test_returns_image_analysis(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert isinstance(result, ImageAnalysis)

    def test_lighting_type_is_valid(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert result.lighting_type in LIGHTING_TYPES

    def test_key_is_valid(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert result.key in ("low", "normal", "high")

    def test_l_percentiles_populated(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert len(result.l_percentiles) == 7
        for p in result.l_percentiles:
            assert 0.0 <= p <= 255.0
        # Percentiles must be non-decreasing
        for i in range(1, len(result.l_percentiles)):
            assert result.l_percentiles[i] >= result.l_percentiles[i - 1]

    def test_dynamic_range_in_range(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert 0.0 <= result.dynamic_range <= 255.0

    def test_noise_level_in_range(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert 0.0 <= result.noise_level <= 1.0

    def test_contrast_in_range(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert 0.0 <= result.contrast_level <= 1.0

    def test_sharpness_in_range(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert 0.0 <= result.sharpness_estimate <= 1.0

    def test_white_balance_estimate_keys(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        wb = result.white_balance_estimate
        assert set(wb.keys()) == {"kelvin", "tint", "cast_strength"}
        assert 2000.0 <= wb["kelvin"] <= 12000.0
        assert -100.0 <= wb["tint"] <= 100.0
        assert 0.0 <= wb["cast_strength"] <= 1.0

    def test_dominant_colors_shape(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert isinstance(result.dominant_colors, list)
        assert 1 <= len(result.dominant_colors) <= 5
        for c in result.dominant_colors:
            assert len(c) == 3
            for v in c:
                assert 0 <= v <= 255

    def test_skin_condition_default_absent(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient())
        assert isinstance(result.skin_condition, SkinCondition)
        assert result.skin_condition.present is False

    def test_skin_condition_populated_with_face_and_mask(self):
        analyzer = ImageAnalyzer()
        img = _skin_tone_block()
        mask = _skin_mask_block()
        result = analyzer.analyze(img, face_bboxes=[(80, 60, 100, 140)], skin_mask=mask)
        assert result.skin_condition.present is True
        assert 0.0 <= result.skin_condition.l_mean <= 255.0
        assert 0.0 <= result.skin_condition.c_std
        assert 0.0 <= result.skin_condition.redness <= 1.0
        assert 0.0 <= result.skin_condition.yellowness <= 1.0
        assert 0.0 <= result.skin_condition.uniformity <= 1.0


# ---------------------------------------------------------------------------
# Tests: different images produce different analyses
# ---------------------------------------------------------------------------


class TestDifferentImagesProduceDifferentAnalyses:
    def test_dark_noisy_vs_bright_flat_differ(self):
        analyzer = ImageAnalyzer()
        dark = analyzer.analyze(_dark_noisy())
        bright = analyzer.analyze(_bright_flat())
        assert dark.mean_luminance < bright.mean_luminance
        assert dark.noise_level > bright.noise_level
        # Bright flat field should be high-key; dark noisy should be low-key
        assert dark.key == "low"
        assert bright.key == "high"

    def test_warm_vs_cool_cast_differ_in_kelvin(self):
        analyzer = ImageAnalyzer()
        warm = analyzer.analyze(_warm_cast())
        cool = analyzer.analyze(_cool_cast())
        assert warm.white_balance_estimate["kelvin"] < cool.white_balance_estimate["kelvin"]

    def test_low_light_classified(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_dark_noisy())
        assert result.lighting_type == "low_light"

    def test_analysis_is_deterministic(self):
        analyzer = ImageAnalyzer()
        a1 = analyzer.analyze(_gradient())
        a2 = analyzer.analyze(_gradient())
        # Same input → same output (k-means uses a fixed seed)
        assert a1.mean_luminance == a2.mean_luminance
        assert a1.lighting_type == a2.lighting_type
        assert a1.dominant_colors == a2.dominant_colors


# ---------------------------------------------------------------------------
# Tests: suggest_params returns valid param names
# ---------------------------------------------------------------------------


class TestSuggestParams:
    def test_returns_dict(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_gradient())
        params = analyzer.suggest_params(analysis)
        assert isinstance(params, dict)

    def test_all_keys_are_valid_param_names(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_dark_noisy())
        params = analyzer.suggest_params(analysis)
        valid_names = set(param_names())
        for k in params:
            assert k in valid_names, f"unknown param name: {k}"

    def test_dark_image_suggests_brightness(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_dark_noisy())
        params = analyzer.suggest_params(analysis)
        # Dark image → positive brightness
        assert "brightness" in params
        assert params["brightness"] > 0

    def test_bright_image_suggests_negative_brightness(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_bright_flat(value=230))
        params = analyzer.suggest_params(analysis)
        if "brightness" in params:
            assert params["brightness"] < 0

    def test_noisy_image_suggests_denoise(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_dark_noisy())
        params = analyzer.suggest_params(analysis)
        assert "ai_denoise" in params
        assert 0 <= params["ai_denoise"] <= 100

    def test_warm_cast_suggests_wb_kelvin(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_warm_cast())
        params = analyzer.suggest_params(analysis)
        # Warm cast → cooler kelvin suggested (below 6500)
        if "white_balance_kelvin" in params:
            assert 2000 <= params["white_balance_kelvin"] <= 12000

    def test_params_within_spec_bounds(self):
        """Every suggested value must respect its ParamSpec min/max."""
        from retouch.params import get_param, ParamSpec
        analyzer = ImageAnalyzer()
        for img in [_gradient(), _dark_noisy(), _bright_flat(), _warm_cast(), _cool_cast()]:
            analysis = analyzer.analyze(img)
            params = analyzer.suggest_params(analysis)
            for name, val in params.items():
                spec: ParamSpec = get_param(name)
                if spec.min_val is not None:
                    assert val >= spec.min_val, f"{name}={val} < min {spec.min_val}"
                if spec.max_val is not None:
                    assert val <= spec.max_val, f"{name}={val} > max {spec.max_val}"


# ---------------------------------------------------------------------------
# Tests: suggest_recipe returns a valid recipe name
# ---------------------------------------------------------------------------


class TestSuggestRecipe:
    def test_returns_string(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_gradient())
        recipe = analyzer.suggest_recipe(analysis)
        assert isinstance(recipe, str)
        assert recipe

    def test_returns_known_recipe(self):
        analyzer = ImageAnalyzer()
        for img in [_gradient(), _dark_noisy(), _bright_flat(), _warm_cast(), _cool_cast()]:
            analysis = analyzer.analyze(img)
            recipe = analyzer.suggest_recipe(analysis)
            assert recipe in RECIPES, f"suggested unknown recipe: {recipe}"

    def test_low_light_suggests_cool_or_soft_recipe(self):
        analyzer = ImageAnalyzer()
        analysis = analyzer.analyze(_dark_noisy())
        recipe = analyzer.suggest_recipe(analysis)
        # Low-light / noisy → milk_skin, moonlight_porcelain, or xhs_ultrasoft
        assert recipe in {"milk_skin_v1", "moonlight_porcelain", "xhs_ultrasoft", "natural"}

    def test_default_fallback_is_valid(self):
        analyzer = ImageAnalyzer()
        # A neutral mid-gray image should fall through to the default
        gray = np.full((128, 128, 3), 128, dtype=np.uint8)
        analysis = analyzer.analyze(gray)
        recipe = analyzer.suggest_recipe(analysis)
        assert recipe in RECIPES


# ---------------------------------------------------------------------------
# Tests: dtype handling
# ---------------------------------------------------------------------------


class TestDtypeHandling:
    def test_uint8_input(self):
        analyzer = ImageAnalyzer()
        img = _gradient()
        assert img.dtype == np.uint8
        result = analyzer.analyze(img)
        assert isinstance(result, ImageAnalysis)

    def test_float32_input(self):
        analyzer = ImageAnalyzer()
        img = _gradient().astype(np.float32)
        assert img.dtype == np.float32
        result = analyzer.analyze(img)
        assert isinstance(result, ImageAnalysis)
        # float32 path must match uint8 path (no dtype-dependent divergence
        # in the measurement math)
        u8_result = analyzer.analyze(_gradient())
        assert abs(result.mean_luminance - u8_result.mean_luminance) < 1.0

    def test_float32_out_of_range_clipped(self):
        analyzer = ImageAnalyzer()
        # float32 with values outside [0, 255] — must clip, not crash
        img = np.full((64, 64, 3), 300.0, dtype=np.float32)
        result = analyzer.analyze(img)
        assert isinstance(result, ImageAnalysis)
        # Clipped to 255 → high-key
        assert result.key == "high"

    def test_rejects_non_3channel(self):
        analyzer = ImageAnalyzer()
        img = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError):
            analyzer.analyze(img)

    def test_rejects_non_3channel_float(self):
        analyzer = ImageAnalyzer()
        img = np.zeros((64, 64, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            analyzer.analyze(img)

    def test_small_image_does_not_crash(self):
        analyzer = ImageAnalyzer()
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        result = analyzer.analyze(img)
        assert isinstance(result, ImageAnalysis)

    def test_skin_mask_uint8_accepted(self):
        """A [0, 255] uint8 mask must be normalized to [0, 1] internally."""
        analyzer = ImageAnalyzer()
        img = _skin_tone_block()
        mask = (_skin_mask_block() * 255).astype(np.uint8)
        result = analyzer.analyze(img, face_bboxes=[(80, 60, 100, 140)], skin_mask=mask)
        assert result.skin_condition.present is True

    def test_skin_mask_resized_to_match(self):
        """A mask of a different shape must be resized internally."""
        analyzer = ImageAnalyzer()
        img = _skin_tone_block(h=256, w=256)
        # Mask at half resolution — must be upsampled
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[30:100, 40:90] = 1.0
        result = analyzer.analyze(img, face_bboxes=[(80, 60, 100, 140)], skin_mask=mask)
        assert result.skin_condition.present is True


# ---------------------------------------------------------------------------
# Tests: small-image / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_downsample_dim_too_small_rejected(self):
        with pytest.raises(ValueError):
            ImageAnalyzer(downsample_dim=32)

    def test_no_face_no_mask_no_crash(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient(), face_bboxes=None, skin_mask=None)
        assert result.skin_condition.present is False

    def test_face_without_mask_no_crash(self):
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(_gradient(), face_bboxes=[(0, 0, 50, 50)], skin_mask=None)
        assert result.skin_condition.present is False

    def test_empty_skin_mask_treated_as_absent(self):
        analyzer = ImageAnalyzer()
        mask = np.zeros((256, 256), dtype=np.float32)
        result = analyzer.analyze(
            _gradient(), face_bboxes=[(0, 0, 50, 50)], skin_mask=mask
        )
        assert result.skin_condition.present is False
