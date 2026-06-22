"""Tests for retouch/engine.py — helper functions, ProcessingContext, ProcessingResult."""

import cv2
import numpy as np
import pytest

from retouch.engine import (
    RetouchEngine,
    ProcessingContext,
    ProcessingResult,
    _FaceResult,
    _CoreResult,
    resolve_recipe,
    _deep_merge,
    build_context,
    _adjust_contrast,
    _adjust_tonal,
    _apply_selective_sharpening,
    _norm_mask,
    _accum,
)
from retouch.recipes import RECIPES


class TestProcessingContext:
    def test_default_values(self):
        ctx = ProcessingContext()
        assert ctx.smooth == 50.0
        assert ctx.whiten == 0.0
        assert ctx.contrast == 0.0
        assert ctx.impact == 0.0
        assert ctx.bloom == 0.0
        assert ctx.bloom_threshold == 210.0
        assert ctx.bloom_softness == 30.0
        assert ctx.active_recipe == "natural"
        assert ctx.whiten_tone == "rosy"

    def test_custom_values(self):
        ctx = ProcessingContext(smooth=80.0, contrast=15.0, active_recipe="cosplay")
        assert ctx.smooth == 80.0
        assert ctx.contrast == 15.0
        assert ctx.active_recipe == "cosplay"

    def test_optional_fields_default_to_none(self):
        ctx = ProcessingContext()
        assert ctx.brightness is None
        assert ctx.lip_tint is None
        assert ctx.color_grade is None
        assert ctx.nose_smooth is None


class TestProcessingResult:
    def test_ndarray_behaviour(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        pr = ProcessingResult(
            image=img,
            face_count=1,
            timings={"detection": 10.0},
        )
        assert pr.shape == (10, 10, 3)
        assert pr.dtype == np.uint8
        assert pr.face_count == 1
        assert pr.timings["detection"] == 10.0
        assert np.all(pr == 128)

    def test_params_accessible(self):
        ctx = ProcessingContext(smooth=75.0)
        img = np.full((5, 5, 3), 100, dtype=np.uint8)
        pr = ProcessingResult(image=img, params=ctx)
        assert pr.params.smooth == 75.0

    def test_no_face_count_default(self):
        img = np.full((5, 5, 3), 100, dtype=np.uint8)
        pr = ProcessingResult(image=img)
        assert pr.face_count == 0
        assert pr.timings == {}


class TestResolveRecipe:
    def test_known_recipe(self):
        for name in RECIPES:
            rec = resolve_recipe(name)
            assert isinstance(rec, dict)
            break

    def test_unknown_falls_back_to_natural(self):
        rec = resolve_recipe("nonexistent_recipe_xyz")
        assert rec is not None
        assert "frequency" in rec

    def test_extends_merges(self):
        if "natural" in RECIPES:
            rec = resolve_recipe("natural")
            assert "frequency" in rec or "skin" in rec

    def test_recursive_extends_merges(self):
        # 'soft' extends 'anime_cinematic_soft', which extends 'anime_cinematic_v1', which extends 'natural'
        rec = resolve_recipe("soft")
        # Should have inherited from natural
        assert "color_harmony" in rec
        assert rec["color_harmony"]["preset"] == "natural"
        # Should have inherited from anime_cinematic_v1
        assert rec["frequency"]["smooth"] == 0.32
        # Should have the override from anime_cinematic_soft
        assert rec["bloom"]["opacity"] == 0.22


class TestDeepMerge:
    def test_empty_overrides(self):
        base = {"a": 1, "b": {"c": 2}}
        result = base.copy()
        _deep_merge(result, {})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        result = base.copy()
        _deep_merge(result, {"a": 10})
        assert result["a"] == 10
        assert result["b"] == 2

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        result = base.copy()
        _deep_merge(result, {"a": {"y": 99, "z": 3}})
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 99
        assert result["a"]["z"] == 3

    def test_preserves_unrelated_keys(self):
        base = {"a": 1, "b": 2}
        result = base.copy()
        _deep_merge(result, {"c": 3})
        assert "a" in result and "b" in result and "c" in result

    def test_new_nested_key(self):
        base = {"a": {"b": 1}}
        result = base.copy()
        _deep_merge(result, {"a": {"c": {"d": 2}}})
        assert result["a"]["b"] == 1
        assert result["a"]["c"]["d"] == 2


class TestFaceResult:
    def test_stores_data(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        fr = _FaceResult(canvas=img, skin_mask=mask, skin_hair_mask=mask, lips_mask=mask, sharpen_mask=mask, roi_box=(0, 0, 10, 10))
        assert fr.canvas is img
        assert fr.roi_box == (0, 0, 10, 10)
        assert fr.skin_mask is mask
        assert fr.lips_mask is mask

    def test_optional_fields_settable(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=np.float32)
        fr = _FaceResult(canvas=img, skin_mask=mask, skin_hair_mask=mask, lips_mask=mask, sharpen_mask=mask, roi_box=(0, 0, 2, 2))
        fr.teeth_mask = np.ones((2, 2), dtype=np.float32)
        assert fr.teeth_mask is not None


class TestCoreResult:
    def test_stores_values(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        cr = _CoreResult(result=img, acc_skin=mask, acc_skin_hair=mask, acc_lips=mask, acc_sharpen=mask, faces=[], person_mask=mask)
        assert cr.result is img
        assert cr.acc_lips is mask
        assert cr.acc_sharpen is mask
        assert cr.faces == []
        assert cr.no_face is False

    def test_no_face_flag(self):
        cr = _CoreResult(result=np.zeros((2, 2, 3), dtype=np.uint8), acc_skin=None, acc_skin_hair=None, acc_lips=None, acc_sharpen=None, faces=[], person_mask=None, no_face=True)
        assert cr.no_face is True


class TestBuildContext:
    def test_returns_processing_context(self):
        rec = RECIPES.get("natural", {})
        ctx = build_context("natural", rec, {})
        assert isinstance(ctx, ProcessingContext)

    def test_override_smooth(self):
        rec = RECIPES.get("natural", {})
        ctx = build_context("natural", rec, {"smooth": 90.0})
        assert ctx.smooth == 90.0

    def test_override_bloom(self):
        rec = {"bloom": {"opacity": 0.05, "threshold": 220.0, "softness": 15.0}}
        ctx = build_context("natural", rec, {"bloom": 40.0})
        # Override wins
        assert ctx.bloom == 40.0
        # Recipe default threshold wins
        assert ctx.bloom_threshold == 220.0
        assert ctx.bloom_softness == 15.0

    def test_color_grade_default_intensity(self):
        rec = RECIPES.get("natural", {})
        ctx = build_context("natural", rec, {"color_grade": "cosplay"})
        assert ctx.grade_intensity == 1.0
        assert ctx.color_grade == "cosplay"

    def test_grade_intensity_override(self):
        rec = RECIPES.get("natural", {})
        ctx = build_context("natural", rec, {"color_grade": "cosplay", "grade_intensity": 0.5})
        assert ctx.grade_intensity == 0.5

    def test_dark_circles_do_not_fall_back_to_whites(self):
        rec = {"eyes": {"whites": 0.8}}
        ctx = build_context("natural", rec, {})
        assert ctx.dark_circles == 0.0
        assert ctx.teeth_whiten == 80.0

    def test_override_modular_flags(self):
        rec = {"nose_blush": False, "under_eye_blush": False, "white_costume_lift": False}
        ctx = build_context("natural", rec, {"nose_blush": True, "under_eye_blush": True, "white_costume_lift": True})
        assert ctx.nose_blush is True
        assert ctx.under_eye_blush is True
        assert ctx.white_costume_lift is True

        rec_true = {"nose_blush": True, "under_eye_blush": True, "white_costume_lift": True}
        ctx_false = build_context("natural", rec_true, {"nose_blush": False, "under_eye_blush": False, "white_costume_lift": False})
        assert ctx_false.nose_blush is False
        assert ctx_false.under_eye_blush is False
        assert ctx_false.white_costume_lift is False



class TestCompositeFaces:
    def test_uses_union_of_edited_masks(self):
        engine = RetouchEngine.__new__(RetouchEngine)
        base = np.zeros((4, 4, 3), dtype=np.uint8)
        canvas = base.copy()
        canvas[1, 1] = [10, 20, 30]
        canvas[1, 2] = [40, 50, 60]
        canvas[2, 1] = [70, 80, 90]

        skin_hair = np.zeros((4, 4), dtype=np.float32)
        lips = np.zeros((4, 4), dtype=np.float32)
        sharpen = np.zeros((4, 4), dtype=np.float32)
        skin_hair[1, 1] = 1.0
        lips[1, 2] = 1.0
        sharpen[2, 1] = 1.0

        face_result = _FaceResult(
            canvas=canvas,
            skin_mask=np.zeros((4, 4), dtype=np.float32),
            skin_hair_mask=skin_hair,
            lips_mask=lips,
            sharpen_mask=sharpen,
            roi_box=(0, 0, 4, 4),
        )

        result, _, _, acc_lips, acc_sharpen = engine._composite_faces(base, [face_result], 4, 4)

        assert np.array_equal(result[1, 1], [10, 20, 30])
        assert np.array_equal(result[1, 2], [40, 50, 60])
        assert np.array_equal(result[2, 1], [70, 80, 90])
        assert acc_lips[1, 2] == 1.0
        assert acc_sharpen[2, 1] == 1.0


class TestAdjustContrast:
    def test_zero_returns_same(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_contrast(img, 0)
        assert np.all(result == img)

    def test_positive_contrast(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _adjust_contrast(img, 50)
        assert not np.allclose(result, img)

    def test_output_type(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _adjust_contrast(img, 30)
        assert result.dtype == np.uint8


class TestAdjustTonal:
    def test_no_adjustment(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_tonal(img)
        assert np.all(result == img)

    def test_shadows(self):
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        result = _adjust_tonal(img, shadows=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() > 50

    def test_highlights(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _adjust_tonal(img, highlights=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() > 200


class TestApplySelectiveSharpening:
    def test_zero_mask(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask)
        assert np.all(result == img)

    def test_full_mask_unchanged_for_flat(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask, amount=1.0)
        assert np.allclose(result, img, atol=1)

    def test_output_shape(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mask = np.random.rand(20, 20).astype(np.float32)
        result = _apply_selective_sharpening(img, mask)
        assert result.shape == (20, 20, 3)

    def test_without_threshold(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mask = np.random.rand(20, 20).astype(np.float32)
        result = _apply_selective_sharpening(img, mask, threshold=0)
        assert result.shape == (20, 20, 3)


class TestNormMask:
    def test_none(self):
        assert _norm_mask(None) is None

    def test_already_float(self):
        m = np.random.rand(10, 10).astype(np.float32)
        result = _norm_mask(m)
        assert result.dtype == np.float32
        assert result.max() <= 1.0

    def test_uint8_conversion(self):
        m = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = _norm_mask(m)
        assert result.dtype == np.float32
        assert result.max() <= 1.0


class TestAccum:
    def test_none_mask(self):
        acc = np.zeros((10, 10), dtype=np.float32)
        result = _accum(acc, None)
        assert np.all(result == acc)

    def test_accumulates(self):
        acc = np.zeros((10, 10), dtype=np.float32)
        m = np.ones((10, 10), dtype=np.float32) * 0.5
        result = _accum(acc, m)
        assert np.allclose(result, 0.5)

    def test_clamps(self):
        acc = np.ones((10, 10), dtype=np.float32)
        m = np.ones((10, 10), dtype=np.float32)
        result = _accum(acc, m)
        assert result.max() == 1.0


class TestEngineBloom:
    def test_no_face_fallback_applies_bloom(self, engine):
        # A dark 100x100 image with a bright square in the center
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[45:55, 45:55] = 255
        
        # When processing an image with no faces, _no_face_fallback is triggered
        # We specify bloom override
        res = engine.process(img, bloom=100.0, bloom_threshold=200.0, bloom_softness=10.0)
        
        # Check that bloom is applied by verifying that outer pixels are brightened
        assert res.shape == (100, 100, 3)
        assert np.any(res[40, 40] > 0)
        assert res.params.bloom == 100.0
        assert res.params.bloom_threshold == 200.0
        assert res.params.bloom_softness == 10.0
