"""Integration tests — full RetouchEngine pipeline and end-to-end processing.

These tests validate that:
- The pipeline runs without crashing on both real and synthetic images.
- The output image has the correct shape, dtype, and channel order.
- The ProcessingResult carries all expected metadata (timings, face_count, etc.).
- Various recipes can be applied without error.
"""

import numpy as np
import cv2
import pytest

from retouch.engine import RetouchEngine, retouch, ProcessingResult
from retouch.recipes import RECIPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_output(result, input_img, expected_faces=None):
    """Assert basic invariants of a ProcessingResult."""
    assert result.shape == input_img.shape, f"Shape mismatch: {result.shape} vs {input_img.shape}"
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert isinstance(result.face_count, int)
    assert isinstance(result.timings, dict)
    if expected_faces is not None:
        assert result.face_count == expected_faces
    # Timings should be populated when a face is detected
    if result.face_count > 0:
        assert "detection" in result.timings
        assert "per_face" in result.timings


# ---------------------------------------------------------------------------
# Pipeline smoke tests  (synthetic face — may or may not be detected)
# ---------------------------------------------------------------------------

class TestPipelineOnSyntheticFace:
    """Run the full pipeline on a synthetic cartoon face.

    MediaPipe may not detect the drawn face, so these tests validate that
    the pipeline handles both the face-found and no-face paths gracefully.
    """

    def test_pipeline_runs_all_recipes(self, engine, synthetic_face):
        for recipe_name in RECIPES:
            result = engine.process(synthetic_face, recipe=recipe_name)
            _assert_valid_output(result, synthetic_face)
            assert result.params.active_recipe == recipe_name

    def test_pipeline_with_overrides(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="cosplay",
            smooth=80.0,
            whiten=50.0,
            contrast=15.0,
            eye_enhance=60.0,
            lip_enhance=40.0,
            impact=30.0,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_fast_preview(self, engine, synthetic_face):
        result = engine.process(synthetic_face, recipe="natural", fast=True)
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_auto_exposure(self, engine, synthetic_face):
        result = engine.process(synthetic_face, recipe="natural", auto_exposure=True)
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_tonal_adjustments(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            brightness=15.0,
            highlights=10.0,
            shadows=-5.0,
            whites=5.0,
            blacks=-3.0,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_color_grade_only(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            color_grade="film",
            grade_intensity=0.8,
            grain=0.05,
            chromatic_aberration=2.0,
            halation=0.2,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_nose_smooth(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            nose_smooth=30.0,
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_specular_bloom(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            specular_bloom=50.0,
            specular_bloom_tone="neutral",
        )
        _assert_valid_output(result, synthetic_face)

    def test_pipeline_whiten_tone(self, engine, synthetic_face):
        for tone in ("rosy", "porcelain", "neutral"):
            result = engine.process(
                synthetic_face,
                recipe="natural",
                whiten=40.0,
                whiten_tone=tone,
            )
            _assert_valid_output(result, synthetic_face)

    def test_pipeline_dodge_burn(self, engine, synthetic_face):
        result = engine.process(
            synthetic_face,
            recipe="natural",
            dodge_burn=50.0,
        )
        _assert_valid_output(result, synthetic_face)


# ---------------------------------------------------------------------------
# No-face fallback
# ---------------------------------------------------------------------------

class TestNoFaceFallback:
    """When no face is detected, the pipeline should still produce output."""

    def test_blank_image_no_crash(self, engine, blank_image):
        result = engine.process(blank_image, recipe="natural")
        _assert_valid_output(result, blank_image, expected_faces=0)

    def test_blank_image_with_color_grade(self, engine, blank_image):
        result = engine.process(
            blank_image,
            recipe="natural",
            color_grade="film",
            grade_intensity=0.7,
        )
        _assert_valid_output(result, blank_image, expected_faces=0)

    def test_blank_image_with_impact(self, engine, blank_image):
        result = engine.process(
            blank_image, recipe="natural", impact=50.0,
        )
        _assert_valid_output(result, blank_image, expected_faces=0)

    def test_blank_image_all_recipes(self, engine, blank_image):
        for recipe_name in RECIPES:
            result = engine.process(blank_image, recipe=recipe_name)
            _assert_valid_output(result, blank_image)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

class TestRetouchFunction:
    """Test the backward-compatible ``retouch()`` function."""

    def test_retouch_basic(self, synthetic_face):
        result = retouch(synthetic_face)
        _assert_valid_output(result, synthetic_face)

    def test_retouch_with_kwargs(self, synthetic_face):
        result = retouch(
            synthetic_face,
            smooth=70,
            whiten=40,
            eye_enhance=50,
            contrast=10,
            preset="cosplay",
        )
        _assert_valid_output(result, synthetic_face)

    def test_retouch_context_manager(self, synthetic_face):
        with RetouchEngine() as engine:
            result = engine.process(synthetic_face, preset="natural")
        _assert_valid_output(result, synthetic_face)

    def test_retouch_returns_processing_result(self, synthetic_face):
        result = retouch(synthetic_face)
        assert isinstance(result, ProcessingResult)


# ---------------------------------------------------------------------------
# Real-image integration  (only if test_output/ images exist)
# ---------------------------------------------------------------------------

class TestWithRealImage:
    """Full pipeline tests with a real human face image.

    These tests are skipped if no image is available (e.g. in CI).
    """

    @pytest.fixture
    def real_face(self, natural_image_path):
        if natural_image_path is None:
            pytest.skip("No real face image found (test_output/ is empty)")
        img = cv2.imread(natural_image_path)
        if img is None:
            pytest.skip(f"Could not read image: {natural_image_path}")
        return img

    def test_detects_face(self, engine, real_face):
        """Verify that at least one face is detected in the real image."""
        faces = engine._detector.detect(real_face)
        assert len(faces) >= 1, "No face detected in real test image"

    def test_full_pipeline_real_face(self, engine, real_face):
        result = engine.process(real_face, recipe="natural")
        _assert_valid_output(result, real_face)
        assert result.face_count >= 1

    def test_multiple_recipes_real_face(self, engine, real_face):
        for recipe in ("natural", "cosplay", "magazine", "beauty", "film"):
            result = engine.process(real_face, recipe=recipe)
            _assert_valid_output(result, real_face)

    def test_real_face_timings_populated(self, engine, real_face):
        result = engine.process(real_face, recipe="natural")
        expected_stages = {
            "detection", "reshape", "per_face",
            "global", "grading", "finish", "total",
        }
        assert expected_stages.issubset(result.timings.keys()), (
            f"Missing timing stages: {expected_stages - set(result.timings.keys())}"
        )
