"""Production-facing coverage for the opt-in P6 clarity gate."""

import inspect

import cv2
import numpy as np
import pytest

from retouch.engine import ProcessingContext, RetouchEngine, build_context
from retouch.grading import ColorGrader
from retouch.params import resolve_recipe


def _gray(values):
    channel = np.asarray(values, dtype=np.float32)
    return np.repeat(channel[..., None], 3, axis=2)


def _hp_rms(image):
    gray = image[..., 0]
    high = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
    return float(np.sqrt(np.mean(high * high)))


def _engine_with_grader():
    engine = RetouchEngine.__new__(RetouchEngine)
    engine._grader = ColorGrader()
    return engine


def test_noise_aware_gate_is_explicit_and_legacy_default_is_unchanged():
    image = np.random.default_rng(42).uniform(.1, .9, (96, 112, 3)).astype(np.float32)
    grader = ColorGrader()
    legacy = grader._F_add_clarity(image, .10)
    engine = _engine_with_grader()

    default = engine._stage_global(image, ProcessingContext(clarity=10))
    np.testing.assert_array_equal(default, legacy)

    gated = engine._stage_global(
        image, ProcessingContext(clarity=10, clarity_noise_aware=True)
    )
    np.testing.assert_allclose(gated, grader._F_add_clarity_noise_aware(image, .10))
    assert np.isfinite(gated).all()
    assert gated.min() >= 0.0 and gated.max() <= 1.0


def test_noise_aware_flat_patch_is_exact_source_bypass():
    image = _gray(np.full((80, 80), .5, dtype=np.float32))
    grader = ColorGrader()
    output = grader._F_add_clarity_noise_aware(image, .10)
    np.testing.assert_array_equal(output, image)


def test_noise_aware_local_abstention_restores_source_outside_structure():
    size = 128
    image = _gray(np.full((size, size), .5, dtype=np.float32))
    image[:, 92:94, :] = .35
    output = ColorGrader()._F_add_clarity_noise_aware(image, .10)
    # The broad flat side is outside the guided edge and below the local
    # variance floor. It must remain byte-for-byte/float-for-float identical;
    # multiplying a zero increment alone would not protect it from LAB drift.
    np.testing.assert_array_equal(output[:, :40], image[:, :40])


def test_noise_aware_gate_reduces_stochastic_gain_but_keeps_structural_response():
    size = 128
    yy, xx = np.mgrid[:size, :size]
    clean = .5 - .08 * np.exp(-((xx - 36 - .15 * yy) / .9) ** 2)
    noisy = clean + np.random.default_rng(11).normal(0, 2 / 255, clean.shape)
    image = _gray(np.clip(noisy, 0, 1))
    grader = ColorGrader()
    legacy = grader._F_add_clarity(image, .10)
    gated = grader._F_add_clarity_noise_aware(image, .10)
    # Gate is intended to reduce the stochastic residual, not to erase all
    # high-frequency structure. Compare the noisy image's added response.
    assert _hp_rms(gated) < _hp_rms(legacy)
    clean_image = _gray(clean)
    gated_clean = grader._F_add_clarity_noise_aware(clean_image, .10)
    assert _hp_rms(gated_clean) > _hp_rms(clean_image)


def test_uint8_noise_aware_contract_and_zero_identity():
    image = np.random.default_rng(5).integers(20, 235, (64, 72, 3), dtype=np.uint8)
    grader = ColorGrader()
    assert grader._add_clarity_noise_aware(image, 0.0) is image
    output = grader._add_clarity_noise_aware(image, .04)
    assert output.dtype == np.uint8
    assert output.shape == image.shape
    assert int(output.min()) >= 0 and int(output.max()) <= 255
    with pytest.raises(ValueError):
        grader._add_clarity_noise_aware(image.astype(np.float32) / 255.0, .04)


def test_noise_aware_float_contract_rejects_nonfinite_or_out_of_range_inputs():
    grader = ColorGrader()
    image = np.full((20, 20, 3), 0.5, dtype=np.float32)
    for bad in (np.nan, np.inf, -0.01, 1.01):
        candidate = image.copy()
        candidate[0, 0, 0] = bad
        with pytest.raises(ValueError):
            grader._F_add_clarity_noise_aware(candidate, .04)
    with pytest.raises(ValueError):
        grader._F_add_clarity_noise_aware(image, np.nan)


def test_build_context_and_public_process_flag_are_opt_in():
    context = build_context(
        "natural", resolve_recipe("natural"),
        {"clarity": 4.0, "clarity_noise_aware": True},
    )
    assert context.clarity == 4.0
    assert context.clarity_noise_aware is True
    assert ProcessingContext().clarity_noise_aware is False
    assert "clarity_noise_aware" in inspect.signature(RetouchEngine.process).parameters

    from retouch.params import RECIPES
    for recipe_name in RECIPES:
        recipe_context = build_context(recipe_name, resolve_recipe(recipe_name), {})
        assert recipe_context.clarity_noise_aware is False
