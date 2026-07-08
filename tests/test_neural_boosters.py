"""Tests for A4 neural booster placeholders (PARKED infrastructure).

Tests verify that placeholder implementations:
1. Accept valid inputs without crashing
2. Return correctly-shaped empty masks
3. Properly validate input dtype/shape
4. Are gated by enabled=False by default
5. Integrate into engine without affecting output (no-op pass-through)

See MASTER_PLAN.md line 112 and retouch/neural_boosters.py for status.
PARKED: Await A1 evidence before real implementation.
"""

import pytest
import numpy as np
from retouch.neural_boosters import (
    NeuralBooster,
    StrayHairSegmenter,
    DefectSegmenter,
)
from retouch.engine import RetouchEngine, ProcessingContext


# ===========================================================================
# Test NeuralBooster protocol
# ===========================================================================


class TestNeuralBoosterProtocol:
    """Verify NeuralBooster abstract interface."""

    def test_neural_booster_is_abstract(self):
        """NeuralBooster cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NeuralBooster()

    def test_neural_booster_detect_required(self):
        """Subclasses must implement detect()."""

        class IncompleteBooster(NeuralBooster):
            pass

        with pytest.raises(TypeError):
            IncompleteBooster()


# ===========================================================================
# Test StrayHairSegmenter (Placeholder)
# ===========================================================================


class TestStrayHairSegmenter:
    """Verify StrayHairSegmenter placeholder behavior."""

    def test_init_disabled_by_default(self):
        """StrayHairSegmenter starts disabled."""
        segmenter = StrayHairSegmenter()
        assert segmenter.enabled is False

    def test_detect_accepts_uint8_input(self):
        """detect() accepts uint8 [0,255] BGR images."""
        segmenter = StrayHairSegmenter()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask = segmenter.detect(img)
        assert mask.dtype == np.uint8
        assert mask.shape == (480, 640)

    def test_detect_accepts_float32_input(self):
        """detect() accepts float32 [0,255] BGR images."""
        segmenter = StrayHairSegmenter()
        img = np.random.uniform(0, 255, (480, 640, 3)).astype(np.float32)
        mask = segmenter.detect(img)
        assert mask.dtype == np.uint8
        assert mask.shape == (480, 640)

    def test_detect_returns_empty_mask_placeholder(self):
        """Placeholder returns all-zero mask (disabled state)."""
        segmenter = StrayHairSegmenter()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask = segmenter.detect(img)
        assert np.all(mask == 0), "Placeholder should return empty mask"

    def test_detect_preserves_input_shape(self):
        """Output mask matches input image spatial dimensions."""
        segmenter = StrayHairSegmenter()
        for h, w in [(100, 150), (480, 640), (1080, 1920)]:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            mask = segmenter.detect(img)
            assert mask.shape == (h, w)

    def test_detect_rejects_invalid_dtype(self):
        """detect() raises ValueError for unsupported dtype."""
        segmenter = StrayHairSegmenter()
        img_int16 = np.random.randint(0, 256, (480, 640, 3), dtype=np.int16)
        with pytest.raises(ValueError, match="uint8 or float32"):
            segmenter.detect(img_int16)

    def test_detect_rejects_invalid_shape_1d(self):
        """detect() raises ValueError for 1D input."""
        segmenter = StrayHairSegmenter()
        img = np.random.randint(0, 256, 1920, dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            segmenter.detect(img)

    def test_detect_rejects_invalid_shape_2d(self):
        """detect() raises ValueError for 2D (grayscale) input."""
        segmenter = StrayHairSegmenter()
        img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            segmenter.detect(img)

    def test_detect_rejects_invalid_channel_count(self):
        """detect() raises ValueError for non-BGR channel counts."""
        segmenter = StrayHairSegmenter()
        # RGBA (4 channels)
        img_rgba = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            segmenter.detect(img_rgba)
        # Mono expanded (2 channels)
        img_2ch = np.random.randint(0, 256, (480, 640, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            segmenter.detect(img_2ch)

    def test_detect_deterministic(self):
        """Placeholder returns consistent empty mask on repeated calls."""
        segmenter = StrayHairSegmenter()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask1 = segmenter.detect(img)
        mask2 = segmenter.detect(img)
        np.testing.assert_array_equal(mask1, mask2)


# ===========================================================================
# Test DefectSegmenter (Placeholder)
# ===========================================================================


class TestDefectSegmenter:
    """Verify DefectSegmenter placeholder behavior."""

    def test_init_disabled_by_default(self):
        """DefectSegmenter starts disabled."""
        segmenter = DefectSegmenter()
        assert segmenter.enabled is False

    def test_detect_accepts_uint8_input(self):
        """detect() accepts uint8 [0,255] BGR images."""
        segmenter = DefectSegmenter()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask = segmenter.detect(img)
        assert mask.dtype == np.uint8
        assert mask.shape == (480, 640)

    def test_detect_accepts_float32_input(self):
        """detect() accepts float32 [0,255] BGR images."""
        segmenter = DefectSegmenter()
        img = np.random.uniform(0, 255, (480, 640, 3)).astype(np.float32)
        mask = segmenter.detect(img)
        assert mask.dtype == np.uint8
        assert mask.shape == (480, 640)

    def test_detect_returns_empty_mask_placeholder(self):
        """Placeholder returns all-zero confidence map (disabled state)."""
        segmenter = DefectSegmenter()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask = segmenter.detect(img)
        assert np.all(mask == 0), "Placeholder should return empty mask"

    def test_detect_preserves_input_shape(self):
        """Output mask matches input image spatial dimensions."""
        segmenter = DefectSegmenter()
        for h, w in [(100, 150), (480, 640), (4160, 6240)]:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            mask = segmenter.detect(img)
            assert mask.shape == (h, w)

    def test_detect_rejects_invalid_dtype(self):
        """detect() raises ValueError for unsupported dtype."""
        segmenter = DefectSegmenter()
        img_int32 = np.random.randint(0, 256, (480, 640, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="uint8 or float32"):
            segmenter.detect(img_int32)

    def test_detect_rejects_invalid_shape_3d_but_wrong_channels(self):
        """detect() raises ValueError for wrong number of channels."""
        segmenter = DefectSegmenter()
        # 5-channel image
        img = np.random.randint(0, 256, (480, 640, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            segmenter.detect(img)

    def test_detect_rejects_invalid_shape_4d(self):
        """detect() raises ValueError for 4D batch input."""
        segmenter = DefectSegmenter()
        img = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            segmenter.detect(img)

    def test_detect_deterministic(self):
        """Placeholder returns consistent empty mask on repeated calls."""
        segmenter = DefectSegmenter()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask1 = segmenter.detect(img)
        mask2 = segmenter.detect(img)
        np.testing.assert_array_equal(mask1, mask2)


# ===========================================================================
# Test Engine Integration (Wiring Verification)
# ===========================================================================


class TestNeuralBoostersEngineIntegration:
    """Verify neural boosters stage wires into engine without affecting output."""

    @pytest.fixture
    def simple_context(self):
        """Create minimal ProcessingContext with neural boosters disabled."""
        ctx = ProcessingContext()
        ctx.neural_stray_hair_boost = 0.0
        ctx.neural_defect_boost = 0.0
        return ctx

    @pytest.fixture
    def simple_image(self):
        """Create a test image (480x640x3 uint8)."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_stage_neural_boosters_exists_in_engine(self):
        """Engine has _stage_neural_boosters method."""
        engine = RetouchEngine()
        assert hasattr(engine, "_stage_neural_boosters")
        assert callable(engine._stage_neural_boosters)

    def test_stage_returns_uint8_when_disabled(self, simple_image, simple_context):
        """Stage returns image unchanged when neural boosters are disabled."""
        engine = RetouchEngine()
        result = engine._stage_neural_boosters(simple_image.copy(), simple_context)
        np.testing.assert_array_equal(result, simple_image)

    def test_stage_returns_float32_when_disabled_float_input(self):
        """Stage handles float32 input correctly when disabled."""
        engine = RetouchEngine()
        ctx = ProcessingContext()
        ctx.neural_stray_hair_boost = 0.0
        ctx.neural_defect_boost = 0.0

        img_float = np.random.uniform(0, 255, (480, 640, 3)).astype(np.float32)
        result = engine._stage_neural_boosters(img_float.copy(), ctx)
        np.testing.assert_array_equal(result, img_float)
        assert result.dtype == np.float32

    def test_stage_no_op_stray_hair_enabled_but_placeholder_disabled(self):
        """Stage accepts stray_hair param but no-ops (placeholder disabled)."""
        engine = RetouchEngine()
        ctx = ProcessingContext()
        ctx.neural_stray_hair_boost = 50.0  # Enabled in context
        ctx.neural_defect_boost = 0.0

        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = engine._stage_neural_boosters(img.copy(), ctx)
        # Should still be no-op because placeholder is disabled
        np.testing.assert_array_equal(result, img)

    def test_stage_no_op_defect_enabled_but_placeholder_disabled(self):
        """Stage accepts defect param but no-ops (placeholder disabled)."""
        engine = RetouchEngine()
        ctx = ProcessingContext()
        ctx.neural_stray_hair_boost = 0.0
        ctx.neural_defect_boost = 75.0  # Enabled in context

        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = engine._stage_neural_boosters(img.copy(), ctx)
        # Should still be no-op because placeholder is disabled
        np.testing.assert_array_equal(result, img)

    def test_stage_with_person_mask_optional(self):
        """Stage accepts optional person_mask parameter."""
        engine = RetouchEngine()
        ctx = ProcessingContext()
        ctx.neural_stray_hair_boost = 0.0
        ctx.neural_defect_boost = 0.0

        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        person_mask = np.ones((480, 640), dtype=np.uint8)

        result = engine._stage_neural_boosters(img.copy(), ctx, person_mask)
        np.testing.assert_array_equal(result, img)

    def test_stage_gates_on_zero_strength(self):
        """Stage returns early if both boosters are at zero."""
        engine = RetouchEngine()
        ctx = ProcessingContext()
        ctx.neural_stray_hair_boost = 0.0
        ctx.neural_defect_boost = 0.0

        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        original = img.copy()
        result = engine._stage_neural_boosters(img, ctx)
        np.testing.assert_array_equal(result, original)

    def test_processing_context_has_neural_fields(self):
        """ProcessingContext has neural_stray_hair_boost and neural_defect_boost."""
        ctx = ProcessingContext()
        assert hasattr(ctx, "neural_stray_hair_boost")
        assert hasattr(ctx, "neural_defect_boost")
        assert ctx.neural_stray_hair_boost == 0.0
        assert ctx.neural_defect_boost == 0.0

    def test_neural_fields_can_be_set(self):
        """ProcessingContext neural fields are mutable."""
        ctx = ProcessingContext()
        ctx.neural_stray_hair_boost = 25.0
        ctx.neural_defect_boost = 50.0
        assert ctx.neural_stray_hair_boost == 25.0
        assert ctx.neural_defect_boost == 50.0


# ===========================================================================
# Test ParamSpec Wiring (Params Module Integration)
# ===========================================================================


class TestNeuralBoostersParamSpecs:
    """Verify neural booster parameters are registered and accessible."""

    def test_neural_params_in_processing_params(self):
        """Neural booster params are in PROCESSING_PARAMS."""
        from retouch.params import PROCESSING_PARAMS

        neural_param_names = {
            "neural_stray_hair_boost",
            "neural_defect_boost",
        }
        registered_names = {p.name for p in PROCESSING_PARAMS}
        assert neural_param_names.issubset(registered_names)

    def test_stray_hair_param_spec_properties(self):
        """neural_stray_hair_boost param has correct properties."""
        from retouch.params import get_param

        spec = get_param("neural_stray_hair_boost")
        assert spec.name == "neural_stray_hair_boost"
        assert spec.cli_flag == "neural-stray-hair-boost"
        assert spec.default == 0
        assert spec.conversion == "recipe_pct"
        assert spec.min_val == 0
        assert spec.max_val == 100

    def test_defect_param_spec_properties(self):
        """neural_defect_boost param has correct properties."""
        from retouch.params import get_param

        spec = get_param("neural_defect_boost")
        assert spec.name == "neural_defect_boost"
        assert spec.cli_flag == "neural-defect-boost"
        assert spec.default == 0
        assert spec.conversion == "recipe_pct"
        assert spec.min_val == 0
        assert spec.max_val == 100

    def test_neural_params_disabled_by_default_in_context(self):
        """Neural params default to 0 (disabled) in ProcessingContext."""
        ctx = ProcessingContext()
        assert ctx.neural_stray_hair_boost == 0.0
        assert ctx.neural_defect_boost == 0.0

    def test_neural_params_can_be_overridden_via_build_context(self):
        """build_context() respects neural param overrides."""
        from retouch.params import resolve_recipe
        from retouch.engine import build_context

        # Use any baseline recipe (they all have neural params at 0)
        recipe = resolve_recipe("natural")
        ctx = build_context("natural", recipe, {"neural_stray_hair_boost": 50})
        assert ctx.neural_stray_hair_boost == 50


# ===========================================================================
# Test Docstring and Metadata
# ===========================================================================


class TestNeuralBoostersMetadata:
    """Verify module metadata and documentation."""

    def test_module_docstring_mentions_parked_status(self):
        """neural_boosters.py docstring clearly states PARKED status."""
        import retouch.neural_boosters as nb

        assert "PARKED" in nb.__doc__
        assert "A1" in nb.__doc__

    def test_neural_booster_classes_have_docstrings(self):
        """All classes have informative docstrings."""
        assert NeuralBooster.__doc__ is not None
        assert StrayHairSegmenter.__doc__ is not None
        assert DefectSegmenter.__doc__ is not None

    def test_detect_method_documented(self):
        """detect() methods have docstrings."""
        assert StrayHairSegmenter.detect.__doc__ is not None
        assert DefectSegmenter.detect.__doc__ is not None

    def test_stage_method_documented(self):
        """Engine._stage_neural_boosters has docstring explaining parked status."""
        engine = RetouchEngine()
        assert engine._stage_neural_boosters.__doc__ is not None
        assert "PARKED" in engine._stage_neural_boosters.__doc__
        assert "A1" in engine._stage_neural_boosters.__doc__
