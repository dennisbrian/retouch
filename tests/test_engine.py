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
        assert ctx.smooth == 30
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
        hair_only = np.zeros((4, 4), dtype=np.float32)
        skin_hair[1, 1] = 1.0
        lips[1, 2] = 1.0
        sharpen[2, 1] = 1.0
        hair_only[0, 1] = 1.0

        face_result = _FaceResult(
            canvas=canvas,
            skin_mask=np.zeros((4, 4), dtype=np.float32),
            skin_hair_mask=skin_hair,
            lips_mask=lips,
            sharpen_mask=sharpen,
            roi_box=(0, 0, 4, 4),
            hair_only_mask=hair_only,
        )

        result, _, _, acc_lips, acc_sharpen, acc_hair_only = engine._composite_faces(
            base, [face_result], 4, 4,
        )

        assert np.array_equal(result[1, 1], [10, 20, 30])
        assert np.array_equal(result[1, 2], [40, 50, 60])
        assert np.array_equal(result[2, 1], [70, 80, 90])
        assert acc_lips[1, 2] == 1.0
        assert acc_sharpen[2, 1] == 1.0
        assert acc_hair_only[0, 1] == 1.0
        assert not hasattr(engine, "_acc_hair_only")

        _, _, _, _, _, next_hair_only = engine._composite_faces(base, [], 4, 4)
        assert float(next_hair_only.max()) == 0.0


class TestApplyWhiteCostumeLift:
    """Unit tests for ``RetouchEngine._apply_white_costume_lift``.

    The method boosts L (and slightly a/b) in LAB space for pixels that
    are simultaneously bright and chromatically neutral *and* outside the
    skin+lip region. With a zero ``grade_intensity`` the lift amount
    collapses to zero; with a full skin+lip mask the white mask is empty
    and the method short-circuits to the input image.
    """

    @staticmethod
    def _engine():
        return RetouchEngine.__new__(RetouchEngine)

    def test_returns_same_shape_and_dtype(self):
        engine = self._engine()
        img = np.full((20, 20, 3), 240, dtype=np.uint8)
        skin_hair = np.zeros((20, 20), dtype=np.float32)
        lips = np.zeros((20, 20), dtype=np.float32)
        result = engine._apply_white_costume_lift(img, skin_hair, lips, 1.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_zero_grade_intensity_returns_input_unchanged(self):
        engine = self._engine()
        img = np.full((20, 20, 3), 240, dtype=np.uint8)
        skin_hair = np.zeros((20, 20), dtype=np.float32)
        lips = np.zeros((20, 20), dtype=np.float32)
        result = engine._apply_white_costume_lift(img, skin_hair, lips, 0.0)
        assert np.array_equal(result, img)

    def test_full_face_mask_short_circuits(self):
        engine = self._engine()
        img = np.full((20, 20, 3), 240, dtype=np.uint8)
        skin_hair = np.ones((20, 20), dtype=np.float32)
        lips = np.ones((20, 20), dtype=np.float32)
        result = engine._apply_white_costume_lift(img, skin_hair, lips, 1.0)
        assert np.array_equal(result, img)

    def test_lifts_white_regions(self):
        engine = self._engine()
        img = np.full((20, 20, 3), 240, dtype=np.uint8)
        skin_hair = np.zeros((20, 20), dtype=np.float32)
        lips = np.zeros((20, 20), dtype=np.float32)
        result = engine._apply_white_costume_lift(img, skin_hair, lips, 1.0)
        lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab_out[:, :, 0].mean() > lab_in[:, :, 0].mean()

    def test_does_not_affect_dark_pixels(self):
        engine = self._engine()
        img = np.full((20, 20, 3), 30, dtype=np.uint8)
        skin_hair = np.zeros((20, 20), dtype=np.float32)
        lips = np.zeros((20, 20), dtype=np.float32)
        result = engine._apply_white_costume_lift(img, skin_hair, lips, 1.0)
        assert np.array_equal(result, img)


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


class TestNoFaceFallbackFloatNative:
    """F1/E2: _no_face_fallback's white balance and master HSL used to
    round-trip through uint8 (to_uint8/to_float) even though both
    white_balance_lch and adjust_hsl_lch are dtype-aware and accept float32
    directly -- an unnecessary quantization step that _stage_grade (the
    face path) already avoided for the same two operations. See
    docs/plans/PLAN_P4_MAKEUP_UNMIX.md Sec 18.
    """

    def test_no_face_white_balance_runs(self, engine):
        img = np.random.RandomState(5).randint(20, 230, (80, 80, 3), dtype=np.uint8)
        res = engine.process(img, white_balance_kelvin=4500, white_balance_tint=15)
        assert res.shape == img.shape
        assert res.dtype == np.uint8
        assert not np.array_equal(res, img)

    def test_no_face_hsl_runs(self, engine):
        img = np.random.RandomState(5).randint(20, 230, (80, 80, 3), dtype=np.uint8)
        res = engine.process(img, hsl_hue_global=20, hsl_sat_global=15, hsl_lum_global=-5)
        assert res.shape == img.shape
        assert res.dtype == np.uint8
        assert not np.array_equal(res, img)

    def test_no_face_white_balance_more_precise_than_uint8_roundtrip(self):
        """The float-native path should introduce less quantization error
        than a to_uint8/to_float boundary round-trip would -- verified
        directly against retouch.grading.ColorGrader.white_balance_lch's
        own float32 vs uint8 branches (isolated from the rest of the
        engine, which has its own additional processing)."""
        from retouch.grading import ColorGrader

        grader = ColorGrader()
        img_u8 = np.random.RandomState(5).randint(20, 230, (80, 80, 3), dtype=np.uint8)

        # Pure round-trip through LCh with no adjustment (identity check):
        # the uint8 path's own cv2 LAB conversion loses sub-integer
        # precision before any white-balance math is even applied.
        from retouch.color_space import (
            bgr_to_lch, lch_to_bgr, bgr_f32_to_lch_f32, lch_f32_to_bgr_f32,
        )
        back_u8 = lch_to_bgr(bgr_to_lch(img_u8))
        back_f32 = np.clip(
            lch_f32_to_bgr_f32(bgr_f32_to_lch_f32(img_u8.astype(np.float32))),
            0, 255,
        ).astype(np.uint8)

        delta_u8 = np.abs(back_u8.astype(np.int16) - img_u8.astype(np.int16)).max()
        delta_f32 = np.abs(back_f32.astype(np.int16) - img_u8.astype(np.int16)).max()
        # float32 round-trip must be at least as precise (usually far more
        # precise) than the uint8 round-trip's own quantization error.
        assert delta_f32 <= delta_u8


class TestProcessInputGuard:
    """process() entry guard: reject malformed input before dtype coercion."""

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(None, id="none"),
            pytest.param("path.jpg", id="string"),
            pytest.param(np.zeros((10, 10), dtype=np.uint8), id="2-dim"),
            pytest.param(np.zeros((0, 0, 3), dtype=np.uint8), id="empty-0-0-3"),
            pytest.param(np.zeros((10, 0, 3), dtype=np.uint8), id="empty-10-0-3"),
            pytest.param(np.zeros((10, 10, 4), dtype=np.uint8), id="4-channels"),
            pytest.param(np.zeros((10, 10, 3, 1), dtype=np.uint8), id="4-dim"),
        ],
    )
    def test_invalid_input_raises(self, engine, bad):
        with pytest.raises(ValueError, match="HxWx3 BGR ndarray"):
            engine.process(bad)

    def test_valid_uint8_passes_guard(self, engine):
        """Guard must not fire on valid input; tiny no-face image is fine —
        the engine returns a ProcessingResult (no-face fallback path)."""
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        res = engine.process(img)
        assert res.shape == (8, 8, 3)

    def test_processing_result_subclass_passes_guard(self, engine):
        """ProcessingResult IS an ndarray subclass — isinstance guard must
        not reject it."""
        img = ProcessingResult(image=np.zeros((8, 8, 3), dtype=np.uint8))
        assert isinstance(img, np.ndarray)
        res = engine.process(img)
        assert res.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# _upscale_core_result — unit tests for the proxy down/up-scaling helper
# ---------------------------------------------------------------------------


class TestUpscaleCoreResult:
    """Unit tests for RetouchEngine._upscale_core_result (line 883 in engine.py).

    The function resizes the BGR result and all accumulated float masks back
    to the pre-proxy resolution. It is a ``@staticmethod`` so no engine
    instance is required for these tests.
    """

    def test_upsizes_result_to_target(self):
        # 1024-px core result, target 2048-px output
        core = _CoreResult(
            result=np.full((1024, 1024, 3), 100, dtype=np.uint8),
            acc_skin=np.zeros((1024, 1024), dtype=np.float32),
            acc_skin_hair=np.zeros((1024, 1024), dtype=np.float32),
            acc_lips=np.zeros((1024, 1024), dtype=np.float32),
            acc_sharpen=np.zeros((1024, 1024), dtype=np.float32),
            faces=[],
            person_mask=None,
        )
        out = RetouchEngine._upscale_core_result(core, 2048, 2048)
        assert out.result.shape == (2048, 2048, 3)
        assert out.result.dtype == np.uint8

    def test_upsizes_all_masks(self):
        # Each accumulated mask should match the target resolution
        core = _CoreResult(
            result=np.full((512, 512, 3), 50, dtype=np.uint8),
            acc_skin=np.ones((512, 512), dtype=np.float32) * 0.3,
            acc_skin_hair=np.ones((512, 512), dtype=np.float32) * 0.5,
            acc_lips=np.ones((512, 512), dtype=np.float32) * 0.2,
            acc_sharpen=np.ones((512, 512), dtype=np.float32) * 0.1,
            faces=[],
            person_mask=np.ones((512, 512), dtype=np.float32),
        )
        out = RetouchEngine._upscale_core_result(core, 2048, 2048)
        assert out.acc_skin.shape == (2048, 2048)
        assert out.acc_skin_hair.shape == (2048, 2048)
        assert out.acc_lips.shape == (2048, 2048)
        assert out.acc_sharpen.shape == (2048, 2048)
        assert out.person_mask.shape == (2048, 2048)

    def test_downscales_when_target_smaller(self):
        # A core result at 2048-px may be upscaled then the result reduced
        # back down — verify the function handles target < current size.
        core = _CoreResult(
            result=np.full((2048, 2048, 3), 200, dtype=np.uint8),
            acc_skin=np.zeros((2048, 2048), dtype=np.float32),
            acc_skin_hair=np.zeros((2048, 2048), dtype=np.float32),
            acc_lips=np.zeros((2048, 2048), dtype=np.float32),
            acc_sharpen=np.zeros((2048, 2048), dtype=np.float32),
            faces=[],
            person_mask=None,
        )
        out = RetouchEngine._upscale_core_result(core, 1024, 1024)
        assert out.result.shape == (1024, 1024, 3)

    def test_none_masks_preserved(self):
        # If accumulated masks are None, the helper must not crash and must
        # leave them as None.
        core = _CoreResult(
            result=np.full((100, 100, 3), 0, dtype=np.uint8),
            acc_skin=None,
            acc_skin_hair=None,
            acc_lips=None,
            acc_sharpen=None,
            faces=[],
            person_mask=None,
        )
        out = RetouchEngine._upscale_core_result(core, 200, 200)
        assert out.acc_skin is None
        assert out.acc_skin_hair is None
        assert out.acc_lips is None
        assert out.acc_sharpen is None
        assert out.person_mask is None
        assert out.result.shape == (200, 200, 3)

    def test_non_square_target(self):
        # The pre-proxy image can be non-square (e.g. 3000x2000 → 2048x1365).
        core = _CoreResult(
            result=np.full((1365, 2048, 3), 80, dtype=np.uint8),
            acc_skin=np.zeros((1365, 2048), dtype=np.float32),
            acc_skin_hair=np.zeros((1365, 2048), dtype=np.float32),
            acc_lips=np.zeros((1365, 2048), dtype=np.float32),
            acc_sharpen=np.zeros((1365, 2048), dtype=np.float32),
            faces=[],
            person_mask=np.ones((1365, 2048), dtype=np.float32),
        )
        out = RetouchEngine._upscale_core_result(core, 2000, 3000)
        assert out.result.shape == (2000, 3000, 3)
        assert out.acc_skin.shape == (2000, 3000)
        assert out.person_mask.shape == (2000, 3000)

    def test_returns_same_core_object(self):
        # The helper is documented as mutating and returning the same core.
        core = _CoreResult(
            result=np.full((64, 64, 3), 10, dtype=np.uint8),
            acc_skin=np.zeros((64, 64), dtype=np.float32),
            acc_skin_hair=np.zeros((64, 64), dtype=np.float32),
            acc_lips=np.zeros((64, 64), dtype=np.float32),
            acc_sharpen=np.zeros((64, 64), dtype=np.float32),
            faces=[],
            person_mask=None,
        )
        out = RetouchEngine._upscale_core_result(core, 128, 128)
        assert out is core


# ---------------------------------------------------------------------------
# _process_face_core — unit tests for the picklable per-face core pipeline
# ---------------------------------------------------------------------------


def _build_synthetic_regions(h, w, skin_value=0.5):
    """Build a fully-populated ``FaceRegions`` for unit-testing the core."""
    from retouch.parsing import FaceRegions

    r = FaceRegions()
    zeros = np.zeros((h, w), dtype=np.float32)
    r.skin = np.full((h, w), skin_value, dtype=np.float32)
    r.hair = zeros.copy()
    r.lips = zeros.copy()
    r.neck = zeros.copy()
    r.left_eye = zeros.copy()
    r.right_eye = zeros.copy()
    r.left_under_eye = zeros.copy()
    r.right_under_eye = zeros.copy()
    r.left_eyebrow = zeros.copy()
    r.right_eyebrow = zeros.copy()
    r.nose = zeros.copy()
    r.face_oval = zeros.copy()
    r.mouth_interior = zeros.copy()
    r.left_iris = zeros.copy()
    r.right_iris = zeros.copy()
    return r


def _build_synthetic_face(ied=30.0, size=100):
    """Build a 478-landmark ``FaceData`` for unit tests."""
    from retouch.detection import _Landmark, _LandmarkCompat, FaceData

    lm_list = []
    for i in range(478):
        # Spread landmarks within a square at the centre of the crop
        lx = 0.5 + ((i % 17) - 8) * 0.005
        ly = 0.5 + ((i % 13) - 6) * 0.005
        lm_list.append(_Landmark(lx, ly, 0.0))
    compat = _LandmarkCompat(lm_list)
    return FaceData(landmarks=compat, bbox=(0, 0, size, size), ied=ied)


class TestProcessFaceCore:
    """Unit tests for ``_process_face_core`` (perf_optimizations.py line 95).

    This is the standalone picklable function called per-face from worker
    processes. It is exercised here directly with synthetic inputs so the
    happy path and the no-op (zero-strength) path are both validated
    without depending on the full engine pipeline.
    """

    def _make_processors(self):
        """Build a full set of lightweight processors (no ONNX)."""
        from retouch.skin import SkinProcessor
        from retouch.blemish import BlemishRemover
        from retouch.eyes import EyeEnhancer
        from retouch.undereye import UnderEyeRepairer
        from retouch.lips import LipEnhancer
        from retouch.teeth import TeethWhitener
        from retouch.makeup import MakeupEngine
        from retouch.hair import HairEnhancer
        from retouch.relight import Relighter
        from retouch.frequency import FrequencySeparator

        return {
            "skin": SkinProcessor(),
            "relighter": Relighter(),
            "blemish": BlemishRemover(),
            "undereye": UnderEyeRepairer(),
            "eyes": EyeEnhancer(),
            "teeth": TeethWhitener(),
            "lips": LipEnhancer(),
            "makeup": MakeupEngine(),
            "hair": HairEnhancer(),
            "frequency": FrequencySeparator(),
        }

    def _all_zero_ctx(self):
        from retouch.engine import ProcessingContext

        return ProcessingContext(
            smooth=0.0,
            whiten=0.0,
            equalize=0.0,
            blemish=0.0,
            nose_smooth=None,
            pore_synthesis=0.0,
            specular_bloom=0.0,
            dodge_burn=0.0,
            relight=0.0,
            eye_enhance=0.0,
            dark_circles=0.0,
            catchlight=0.0,
            lip_enhance=0.0,
            teeth_whiten=0.0,
            blush=0.0,
            hair_enhance=0.0,
            slimming=0.0,
        )

    def test_returns_face_result_with_expected_attributes(self):
        from retouch.perf_optimizations import _process_face_core, _FaceResult

        roi = 256
        canvas = np.full((roi, roi, 3), 180, dtype=np.uint8)
        regions = _build_synthetic_regions(roi, roi, skin_value=0.5)
        face = _build_synthetic_face(ied=30.0, size=100)
        ctx = self._all_zero_ctx()
        person_mask = np.ones((roi, roi), dtype=np.float32)
        processors = self._make_processors()

        fr = _process_face_core(
            canvas, regions, face, ctx,
            0, 0, roi, roi, person_mask, processors,
        )

        assert isinstance(fr, _FaceResult)
        assert fr.canvas.shape == (roi, roi, 3)
        assert fr.canvas.dtype == np.uint8
        assert fr.skin_mask.shape == (roi, roi)
        assert fr.skin_hair_mask.shape == (roi, roi)
        assert fr.lips_mask.shape == (roi, roi)
        assert fr.sharpen_mask.shape == (roi, roi)
        # roi_box is in (x1, y1, x2, y2) format
        x1, y1, x2, y2 = fr.roi_box
        assert x2 - x1 == roi
        assert y2 - y1 == roi

    def test_masks_are_float32_in_unit_range(self):
        from retouch.perf_optimizations import _process_face_core

        roi = 128
        canvas = np.full((roi, roi, 3), 100, dtype=np.uint8)
        regions = _build_synthetic_regions(roi, roi, skin_value=0.4)
        face = _build_synthetic_face(ied=20.0, size=80)
        ctx = self._all_zero_ctx()
        person_mask = np.ones((roi, roi), dtype=np.float32)
        processors = self._make_processors()

        fr = _process_face_core(
            canvas, regions, face, ctx,
            0, 0, roi, roi, person_mask, processors,
        )

        for m in (fr.skin_mask, fr.skin_hair_mask, fr.lips_mask, fr.sharpen_mask):
            assert m.dtype == np.float32
            assert m.min() >= 0.0
            assert m.max() <= 1.0

    def test_no_op_with_all_zeros_preserves_dimensions(self):
        """With every strength=0 the function must be a near no-op on the
        canvas dimensions, even if pixel values may shift slightly due to
        frequency decomposition."""
        from retouch.perf_optimizations import _process_face_core

        roi = 200
        canvas = np.full((roi, roi, 3), 200, dtype=np.uint8)
        regions = _build_synthetic_regions(roi, roi, skin_value=0.0)
        face = _build_synthetic_face(ied=25.0, size=80)
        ctx = self._all_zero_ctx()
        person_mask = np.ones((roi, roi), dtype=np.float32)
        processors = self._make_processors()

        fr = _process_face_core(
            canvas, regions, face, ctx,
            0, 0, roi, roi, person_mask, processors,
        )

        assert fr.canvas.shape == canvas.shape
        assert fr.skin_mask.max() == 0.0  # zero skin ⇒ zero acc_skin

    def test_safe_auto_review_preserves_parameter_driven_face_render(self):
        """Unavailable detector confidence may request review, but must not
        erase a recipe/explicit face edit that is not an inferred auto delta."""
        import dataclasses

        from retouch.perf_optimizations import _process_face_core

        roi = 128
        canvas = np.full((roi, roi, 3), 120, dtype=np.uint8)
        regions = _build_synthetic_regions(roi, roi, skin_value=0.5)
        face = _build_synthetic_face(ied=24.0, size=80)
        face.confidence = 1.0
        face.confidence_source = "mediapipe_presence_unavailable"
        safe_ctx = self._all_zero_ctx()
        safe_ctx.whiten = 35.0
        safe_ctx._parameter_provenance = {
            "recipe": "natural",
            "explicit_overrides": ["whiten"],
        }
        unsafe_ctx = dataclasses.replace(safe_ctx, safe_auto=False)

        safe = _process_face_core(
            canvas, regions, face, safe_ctx,
            0, 0, roi, roi, np.ones((roi, roi), dtype=np.float32),
            self._make_processors(),
        )
        ungated = _process_face_core(
            canvas, regions, face, unsafe_ctx,
            0, 0, roi, roi, np.ones((roi, roi), dtype=np.float32),
            self._make_processors(),
        )

        np.testing.assert_array_equal(safe.canvas, ungated.canvas)
        assert safe.safe_auto_decisions[-1]["action"] == "review"
        evidence = safe.safe_auto_decisions[-1]["evidence"]
        assert evidence["detector_confidence"]["confidence_measured"] is False
        assert evidence["enforcement"]["parameter_driven_pixels_preserved"] is True

    def test_safe_auto_skip_and_dampen_preserve_parameter_driven_face_render(self):
        """skip (confidence 0.0) and dampen (confidence 0.70) decisions are
        evidence-only: rendered pixels must be byte-identical to an un-gated
        run, never reverted or half-blended toward the pre-edit source."""
        import dataclasses

        from retouch.perf_optimizations import _process_face_core

        roi = 128
        canvas = np.full((roi, roi, 3), 120, dtype=np.uint8)
        regions = _build_synthetic_regions(roi, roi, skin_value=0.5)
        person_mask = np.ones((roi, roi), dtype=np.float32)
        processors = self._make_processors()

        for confidence, expected_action in ((0.0, "skip"), (0.70, "dampen")):
            face = _build_synthetic_face(ied=24.0, size=80)
            face.confidence = confidence
            face.confidence_source = "retinaface"
            safe_ctx = self._all_zero_ctx()
            safe_ctx.whiten = 35.0
            safe_ctx._parameter_provenance = {
                "recipe": "natural",
                "explicit_overrides": ["whiten"],
            }
            unsafe_ctx = dataclasses.replace(safe_ctx, safe_auto=False)

            safe = _process_face_core(
                canvas, regions, face, safe_ctx,
                0, 0, roi, roi, person_mask, processors,
            )
            ungated = _process_face_core(
                canvas, regions, face, unsafe_ctx,
                0, 0, roi, roi, person_mask, processors,
            )

            assert safe.safe_auto_decisions[-1]["action"] == expected_action
            assert (
                safe.safe_auto_decisions[-1]["evidence"]["enforcement"][
                    "parameter_driven_pixels_preserved"
                ]
                is True
            )
            np.testing.assert_array_equal(safe.canvas, ungated.canvas)
            # The edit must actually differ from the pre-edit source, otherwise
            # "preserved" would be vacuous.
            assert not np.array_equal(safe.canvas, canvas)

    def test_function_is_module_level_and_picklable(self):
        """_process_face_core must live at module scope (not as a method on
        a class holding non-picklable ONNX sessions) so workers can call
        it via ``multiprocessing``."""
        import pickle

        from retouch.perf_optimizations import _process_face_core

        # Pickle the function reference itself
        blob = pickle.dumps(_process_face_core)
        restored = pickle.loads(blob)
        assert restored is _process_face_core

    def test_roi_offset_is_reflected_in_roi_box(self):
        """The roi_box returned should reflect the (roi_x1, roi_y1) shift."""
        from retouch.perf_optimizations import _process_face_core

        roi = 128
        canvas = np.full((roi, roi, 3), 150, dtype=np.uint8)
        regions = _build_synthetic_regions(roi, roi)
        face = _build_synthetic_face(ied=20.0, size=60)
        ctx = self._all_zero_ctx()
        person_mask = np.ones((roi, roi), dtype=np.float32)
        processors = self._make_processors()

        # Place ROI at (100, 200) in the full image
        fr = _process_face_core(
            canvas, regions, face, ctx,
            100, 200, roi, roi, person_mask, processors,
        )
        x1, y1, x2, y2 = fr.roi_box
        assert (x1, y1) == (100, 200)
        assert (x2 - x1, y2 - y1) == (roi, roi)
