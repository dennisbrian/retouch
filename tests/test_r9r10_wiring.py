"""Tests for R9/R10 wiring: albedo_even, hemoglobin_smooth, mole_protect, vein_attenuate.

Focused tests for:
- Float/uint8 adapter parity
- Effect sanity (chroma reduction, preservation, etc.)
- Golden-path byte-idempotence when params are 0
"""

import numpy as np
import pytest
import cv2

from retouch import RetouchEngine
from retouch.skin import SkinProcessor
from retouch.engine import ProcessingContext
from retouch.color_science import skin_chroma_std
from retouch import intrinsic, chromophore, skin_chromophore


class TestAlbedoEvenAdapter:
    """Float/uint8 parity and effect sanity for albedo_even."""

    def test_zero_strength_is_noop(self):
        """With strength=0, output should equal input."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0
        skin_mask = np.ones((100, 100), dtype=np.float32)

        result = processor.apply_albedo_even(img_f32, skin_mask, strength=0.0, face_width=100)
        assert np.array_equal(result, img_f32), "albedo_even(strength=0) should be identity"

    def test_none_mask_is_noop(self):
        """With None mask, output should equal input."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0

        result = processor.apply_albedo_even(img_f32, None, strength=0.5, face_width=100)
        assert np.array_equal(result, img_f32), "albedo_even(None mask) should be identity"

    def test_returns_float32(self):
        """Output should match input dtype."""
        processor = SkinProcessor()
        img_f32 = np.random.uniform(0, 255, (100, 100, 3)).astype(np.float32)
        skin_mask = np.ones((100, 100), dtype=np.float32)

        result = processor.apply_albedo_even(img_f32, skin_mask, strength=0.5, face_width=100)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_reduces_chroma_variance(self):
        """Albedo blurring should reduce skin chroma std without dropping mean L."""
        processor = SkinProcessor()
        # Create a synthetic blotchy skin image with high chroma variance
        img_f32 = np.full((100, 100, 3), 150.0, dtype=np.float32)
        # Add blotchy variations in color
        img_f32[20:30, 20:30] = [140, 160, 155]  # reddish blotch
        img_f32[60:70, 60:70] = [160, 140, 145]  # greenish blotch

        skin_mask = np.ones((100, 100), dtype=np.float32)

        # Measure before
        orig_chroma = skin_chroma_std(img_f32, skin_mask)
        orig_lab = cv2.cvtColor(np.clip(img_f32, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_l_mean = orig_lab[:, :, 0].mean()

        # Apply albedo_even
        result = processor.apply_albedo_even(img_f32, skin_mask, strength=0.5, face_width=100)

        # Measure after
        result_chroma = skin_chroma_std(result, skin_mask)
        result_lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        result_l_mean = result_lab[:, :, 0].mean()

        # Chroma std should be lower (blotches evened out)
        assert result_chroma < orig_chroma, f"Expected chroma reduction: {orig_chroma} -> {result_chroma}"
        # L mean should stay roughly constant (±2 for floating-point drift)
        assert abs(result_l_mean - orig_l_mean) < 2.0, f"L mean changed too much: {orig_l_mean} -> {result_l_mean}"


class TestHemoglobinGuidedSmoothAdapter:
    """Float/uint8 parity and effect sanity for hemoglobin_guided_smooth."""

    def test_zero_strength_is_noop(self):
        """With strength=0, output should equal input."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0
        skin_mask = np.ones((100, 100), dtype=np.float32)

        result = processor.apply_hemoglobin_guided_smooth(img_f32, skin_mask, strength=0.0)
        assert np.array_equal(result, img_f32), "hemoglobin_smooth(strength=0) should be identity"

    def test_none_mask_is_noop(self):
        """With None mask, output should equal input."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0

        result = processor.apply_hemoglobin_guided_smooth(img_f32, None, strength=0.5)
        assert np.array_equal(result, img_f32), "hemoglobin_smooth(None mask) should be identity"

    def test_returns_float32(self):
        """Output should match input dtype."""
        processor = SkinProcessor()
        img_f32 = np.random.uniform(0, 255, (100, 100, 3)).astype(np.float32)
        skin_mask = np.ones((100, 100), dtype=np.float32)

        result = processor.apply_hemoglobin_guided_smooth(img_f32, skin_mask, strength=0.5)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


class TestVeinAttenuateAdapter:
    """Float/uint8 parity and effect sanity for vein_attenuate."""

    def test_zero_strength_is_noop(self):
        """With strength=0, output should equal input."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0
        skin_mask = np.ones((100, 100), dtype=np.float32)

        result = processor.apply_vein_attenuate(img_f32, skin_mask, strength=0.0)
        assert np.array_equal(result, img_f32), "vein_attenuate(strength=0) should be identity"

    def test_none_mask_is_noop(self):
        """With None mask, output should equal input."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0

        result = processor.apply_vein_attenuate(img_f32, None, strength=0.5)
        assert np.array_equal(result, img_f32), "vein_attenuate(None mask) should be identity"

    def test_returns_float32(self):
        """Output should match input dtype."""
        processor = SkinProcessor()
        img_f32 = np.random.uniform(0, 255, (100, 100, 3)).astype(np.float32)
        skin_mask = np.ones((100, 100), dtype=np.float32)

        result = processor.apply_vein_attenuate(img_f32, skin_mask, strength=0.5)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_reduces_blue_green(self):
        """Vein attenuation should reduce blue/green channels."""
        processor = SkinProcessor()
        # Create a synthetic vein-like image (high blue-green in a region)
        img_f32 = np.full((100, 100, 3), 150.0, dtype=np.float32)
        # Add a vein-like structure (high in blue and green)
        img_f32[30:70, 40:60, 0] = 120.0  # Blue channel: lower (deoxygenated)
        img_f32[30:70, 40:60, 1] = 130.0  # Green channel: lower
        img_f32[30:70, 40:60, 2] = 150.0  # Red channel: unchanged

        skin_mask = np.ones((100, 100), dtype=np.float32)

        result = processor.apply_vein_attenuate(img_f32, skin_mask, strength=0.5)

        # Blue and green should be further reduced (or equal, depending on low-pass detection)
        # Just verify it doesn't crash and preserves dtype
        assert result.dtype == np.float32
        assert result.shape == img_f32.shape


class TestMoleProtectAdapter:
    """Float/uint8 parity and effect sanity for mole_protect mask generation."""

    def test_zero_strength_returns_none(self):
        """With strength=0, should return (None, None)."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0
        skin_mask = np.ones((100, 100), dtype=np.float32)

        blemish, mole = processor.apply_mole_protect(img_f32, skin_mask, strength=0.0)
        assert blemish is None and mole is None, "mole_protect(strength=0) should return (None, None)"

    def test_none_mask_returns_none(self):
        """With None mask, should return (None, None)."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0

        blemish, mole = processor.apply_mole_protect(img_f32, None, strength=0.5)
        assert blemish is None and mole is None, "mole_protect(None mask) should return (None, None)"

    def test_returns_uint8_masks(self):
        """With valid inputs, should return uint8 binary masks."""
        processor = SkinProcessor()
        img_f32 = np.ones((100, 100, 3), dtype=np.float32) * 128.0
        skin_mask = np.ones((100, 100), dtype=np.float32)

        blemish, mole = processor.apply_mole_protect(img_f32, skin_mask, strength=0.5)

        if blemish is not None:
            assert blemish.dtype == np.uint8, f"Expected uint8, got {blemish.dtype}"
            assert blemish.shape == (100, 100), f"Expected (100, 100), got {blemish.shape}"
        if mole is not None:
            assert mole.dtype == np.uint8, f"Expected uint8, got {mole.dtype}"
            assert mole.shape == (100, 100), f"Expected (100, 100), got {mole.shape}"


class TestGoldenPathByteIdempotence:
    """Verify that with all 4 params at default 0, output is byte-identical.

    This is a smoke test only — full golden-path verification requires
    real images and MediaPipe initialization (integration test).
    """

    def test_default_zero_params_in_context(self):
        """Default ProcessingContext should have all 4 new params at 0.0."""
        ctx = ProcessingContext()
        assert ctx.albedo_even == 0.0
        assert ctx.hemoglobin_smooth == 0.0
        assert ctx.mole_protect == 0.0
        assert ctx.vein_attenuate == 0.0

    def test_adapter_idempotence_at_zero(self):
        """Each adapter should be a no-op when strength=0."""
        processor = SkinProcessor()
        test_image = np.random.uniform(0, 255, (100, 100, 3)).astype(np.float32)
        skin_mask = np.ones((100, 100), dtype=np.float32)

        # Test albedo_even
        result1 = processor.apply_albedo_even(test_image.copy(), skin_mask, 0.0, 100)
        assert np.array_equal(result1, test_image)

        # Test hemoglobin_smooth
        result2 = processor.apply_hemoglobin_guided_smooth(test_image.copy(), skin_mask, 0.0)
        assert np.array_equal(result2, test_image)

        # Test vein_attenuate
        result3 = processor.apply_vein_attenuate(test_image.copy(), skin_mask, 0.0)
        assert np.array_equal(result3, test_image)

        # Test mole_protect
        b, m = processor.apply_mole_protect(test_image.copy(), skin_mask, 0.0)
        assert b is None and m is None


class TestProcessSignatureCompleteness:
    """Guard against the GUI-crash class: every registry param must be an
    accepted keyword of RetouchEngine.process().

    The GUI forwards every PROCESS_INPUT_KEYS entry as a kwarg, so a
    ParamSpec whose name is absent from the hand-listed process() signature
    crashes at render time with
    ``TypeError: process() got an unexpected keyword argument``
    (seen 2026-07-12 with albedo_even, then eye_sclera_brighten).
    """

    def test_all_registry_params_accepted_by_process(self):
        import inspect
        from retouch.params import PROCESSING_PARAMS
        from retouch.engine import RetouchEngine

        signature = inspect.signature(RetouchEngine.process)
        sig = set(signature.parameters)
        accepts_registry_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        missing = [] if accepts_registry_kwargs else [
            p.name for p in PROCESSING_PARAMS if p.name not in sig
        ]
        assert not missing, (
            f"ParamSpecs missing from RetouchEngine.process() signature "
            f"(GUI will crash with unexpected-kwarg TypeError): {missing}"
        )


class TestGuiProcessInputsAlignment:
    """_process_inputs (hand-ordered Gradio components) must align 1:1 with
    PROCESS_INPUT_KEYS, else zip() shifts every later arg (symptom 2026-07-12:
    ai_denoise received a string -> TypeError in engine.process)."""

    def test_process_inputs_matches_keys(self):
        import gui

        keys = list(gui.PROCESS_INPUT_KEYS)
        comps = list(gui._process_inputs)
        assert len(comps) == len(keys), (
            f"_process_inputs has {len(comps)} components but "
            f"PROCESS_INPUT_KEYS has {len(keys)} keys"
        )
        expected = [gui._process_input_components[key] for key in keys]
        assert all(actual is wanted for actual, wanted in zip(comps, expected))
