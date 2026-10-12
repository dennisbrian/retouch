"""Tests for retouch/smart_default.py — SmartProcessor (F10)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retouch.image_analyzer import ImageAnalysis, SkinCondition
from retouch.smart_default import SmartProcessor, SmartSuggestion
from retouch.recipes import RECIPES


# ---------------------------------------------------------------------------
# Synthetic image fixtures (mirrors test_image_analyzer.py)
# ---------------------------------------------------------------------------


def _gradient(h: int = 256, w: int = 256) -> np.ndarray:
    """Neutral grayscale gradient (no WB cast, low noise, normal key)."""
    yy = np.linspace(0, 255, w, dtype=np.float32)
    xx = np.linspace(0, 255, h, dtype=np.float32)
    grid = np.minimum(xx[:, None], yy[None, :])
    img = np.stack([grid, grid, grid], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _dark_noisy(h: int = 256, w: int = 256) -> np.ndarray:
    """Dark, noisy image — low-light scene."""
    rng = np.random.default_rng(42)
    base = np.full((h, w, 3), 40, dtype=np.float32)
    noise = rng.normal(0.0, 18.0, size=(h, w, 3)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


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


def _bright_flat(h: int = 256, w: int = 256, value: int = 220) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _skin_mask_block(h: int = 256, w: int = 256) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    mask[60:200, 80:180] = 1.0
    return mask


def _skin_tone_block(h: int = 256, w: int = 256) -> np.ndarray:
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    img[60:200, 80:180] = (140, 170, 200)
    return img


# ---------------------------------------------------------------------------
# Tests: analyze_and_suggest
# ---------------------------------------------------------------------------


class TestAnalyzeAndSuggest:
    def test_returns_smart_suggestion(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        assert isinstance(result, SmartSuggestion)

    def test_recipe_is_valid(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        assert result.recipe in RECIPES

    def test_analysis_is_image_analysis(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        assert isinstance(result.analysis, ImageAnalysis)

    def test_explanations_is_list_of_str(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        assert isinstance(result.explanations, list)
        assert len(result.explanations) >= 1
        for e in result.explanations:
            assert isinstance(e, str)
            assert len(e) > 0

    def test_params_is_dict(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        assert isinstance(result.params, dict)

    def test_first_explanation_mentions_recipe(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        assert "recipe=" in result.explanations[0]

    def test_dark_noisy_suggests_denoise(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_dark_noisy())
        assert "ai_denoise" in result.params
        denoise_exp = [
            e for e in result.explanations if "ai_denoise" in e
        ]
        assert len(denoise_exp) == 1
        assert "noise" in denoise_exp[0].lower()

    def test_warm_cast_suggests_wb_kelvin(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_warm_cast())
        # Warm cast (high R, low B) → cooler kelvin estimate
        assert "white_balance_kelvin" in result.params
        wb_exp = [
            e for e in result.explanations if "white_balance_kelvin" in e
        ]
        assert len(wb_exp) == 1

    def test_bright_image_reduces_brightness(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_bright_flat(value=230))
        # Mean luminance ~230 > target 128 → negative brightness
        if "brightness" in result.params:
            assert result.params["brightness"] < 0
            bright_exp = [
                e for e in result.explanations if "brightness" in e
            ]
            assert len(bright_exp) == 1

    def test_dark_image_lifts_brightness(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_dark_noisy())
        if "brightness" in result.params:
            assert result.params["brightness"] > 0

    def test_explanations_cover_emitted_params(self):
        """Every non-recipe explanation references a param key that exists."""
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_dark_noisy())
        # The recipe explanation is first; subsequent ones should reference
        # a param key from result.params.
        for exp in result.explanations[1:]:
            # Find a param name mentioned in the explanation
            found = False
            for key in result.params:
                if key in exp:
                    found = True
                    break
            # shadows/clarity/vibrance are lighting-type explanations that
            # reference their own param keys, so this should always hold.
            assert found, f"Explanation references no param: {exp!r}"

    def test_skin_analysis_passed_through(self):
        sp = SmartProcessor()
        img = _skin_tone_block()
        mask = _skin_mask_block()
        result = sp.analyze_and_suggest(
            img, face_bboxes=[(80, 60, 100, 140)], skin_mask=mask
        )
        assert result.analysis is not None
        assert result.analysis.skin_condition.present is True

    def test_uses_injected_analyzer(self):
        """A custom analyzer instance is used, not re-created."""
        from retouch.image_analyzer import ImageAnalyzer
        custom = ImageAnalyzer(downsample_dim=256)
        sp = SmartProcessor(analyzer=custom)
        assert sp._analyzer is custom


# ---------------------------------------------------------------------------
# Tests: process_smart (with mocked engine)
# ---------------------------------------------------------------------------


class TestProcessSmart:
    def test_calls_engine_process_with_recipe(self):
        mock_engine = MagicMock()
        fake_result = np.full((64, 64, 3), 200, dtype=np.uint8)
        mock_engine.process.return_value = fake_result
        sp = SmartProcessor(engine=mock_engine)
        img = _gradient(64, 64)
        out = sp.process_smart(img)
        assert mock_engine.process.called
        call_kwargs = mock_engine.process.call_args
        assert call_kwargs.kwargs.get("recipe") in RECIPES

    def test_returns_engine_result(self):
        mock_engine = MagicMock()
        fake_result = np.full((64, 64, 3), 200, dtype=np.uint8)
        mock_engine.process.return_value = fake_result
        sp = SmartProcessor(engine=mock_engine)
        out = sp.process_smart(_gradient(64, 64))
        assert out is fake_result

    def test_passes_suggested_params_as_kwargs(self):
        mock_engine = MagicMock()
        mock_engine.process.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        sp = SmartProcessor(engine=mock_engine)
        sp.process_smart(_dark_noisy(64, 64))
        call_kwargs = mock_engine.process.call_args.kwargs
        # Dark noisy image should produce ai_denoise suggestion
        if "ai_denoise" in call_kwargs:
            assert call_kwargs["ai_denoise"] is not None

    def test_extra_engine_kwargs_merged(self):
        mock_engine = MagicMock()
        mock_engine.process.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        sp = SmartProcessor(engine=mock_engine)
        sp.process_smart(
            _gradient(64, 64),
            extra_engine_kwargs={"fast": True, "quality": "draft"},
        )
        call_kwargs = mock_engine.process.call_args.kwargs
        assert call_kwargs.get("fast") is True
        assert call_kwargs.get("quality") == "draft"

    def test_extra_kwargs_override_suggestion(self):
        mock_engine = MagicMock()
        mock_engine.process.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        sp = SmartProcessor(engine=mock_engine)
        sp.process_smart(
            _dark_noisy(64, 64),
            extra_engine_kwargs={"ai_denoise": 99},
        )
        call_kwargs = mock_engine.process.call_args.kwargs
        assert call_kwargs.get("ai_denoise") == 99


# ---------------------------------------------------------------------------
# Tests: explanation quality
# ---------------------------------------------------------------------------


class TestExplanations:
    def test_recipe_explanation_includes_lighting(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        first = result.explanations[0]
        assert "lighting=" in first

    def test_recipe_explanation_includes_key(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_gradient())
        first = result.explanations[0]
        assert "key=" in first

    def test_wb_explanation_includes_cast_direction(self):
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_warm_cast())
        wb_exps = [e for e in result.explanations if "white_balance_kelvin" in e]
        if wb_exps:
            assert "warm" in wb_exps[0].lower() or "cool" in wb_exps[0].lower()

    def test_no_duplicate_param_explanations(self):
        """Each param key appears in at most one explanation."""
        sp = SmartProcessor()
        result = sp.analyze_and_suggest(_dark_noisy())
        param_keys_seen: List[str] = []
        for exp in result.explanations[1:]:  # skip recipe explanation
            for key in result.params:
                if key in exp:
                    param_keys_seen.append(key)
        # Allow a param to appear once; duplicates would indicate a bug.
        # (shadows/clarity share the "low-light" reason but are separate keys.)
        assert len(param_keys_seen) == len(set(param_keys_seen)), (
            f"Duplicate param explanations: {param_keys_seen}"
        )


# ---------------------------------------------------------------------------
# Tests: dtype / boundary handling
# ---------------------------------------------------------------------------


class TestBoundaryHandling:
    def test_accepts_uint8_input(self):
        sp = SmartProcessor()
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = sp.analyze_and_suggest(img)
        assert isinstance(result, SmartSuggestion)

    def test_accepts_float32_input(self):
        sp = SmartProcessor()
        img = np.full((64, 64, 3), 128.0, dtype=np.float32)
        result = sp.analyze_and_suggest(img)
        assert isinstance(result, SmartSuggestion)

    def test_rejects_grayscale(self):
        sp = SmartProcessor()
        img = np.full((64, 64), 128, dtype=np.uint8)
        with pytest.raises(ValueError):
            sp.analyze_and_suggest(img)

    def test_rejects_wrong_channels(self):
        sp = SmartProcessor()
        img = np.full((64, 64, 4), 128, dtype=np.uint8)
        with pytest.raises(ValueError):
            sp.analyze_and_suggest(img)


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_image_same_suggestion(self):
        sp = SmartProcessor()
        img = _dark_noisy()
        r1 = sp.analyze_and_suggest(img)
        r2 = sp.analyze_and_suggest(img)
        assert r1.recipe == r2.recipe
        assert r1.params == r2.params
        assert r1.explanations == r2.explanations
