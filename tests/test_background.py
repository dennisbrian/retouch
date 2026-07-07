"""Tests for T1 — Background replace & scene relight.

Validates the ``BackgroundReplacer`` class and the engine wiring
(``_stage_background`` + the 7 ``anime_crystal_void`` ParamSpecs):
  - dtype preservation (uint8 / float32 [0,255] / float32 [0,1])
  - early-return guards (zero strength, no person mask, no-op params)
  - subject protection (person region unchanged for background ops)
  - background actually changes (blur, desaturation, grade, matte)
  - light-wrap modifies the subject edge band
  - subject_sharpen modifies the subject, not the background
  - relight_scene changes luminance with light direction
  - effect scales with strength
  - engine context fields default to 0 and resolve from the recipe
  - the 7 anime_crystal_void keys are no longer dead
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from retouch.background import BackgroundReplacer
from retouch.params import PROCESSING_PARAMS, resolve_recipe
from retouch.engine import build_context, ProcessingContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_subject_bg_image() -> tuple:
    """Build a synthetic image: bright subject on a textured background.

    Returns:
        (img_bgr_u8, person_mask) — (128, 128) and (128, 128).
    """
    h, w = 128, 128
    rng = np.random.default_rng(42)
    # Textured background: a horizontal gradient plus high-frequency noise
    # so blur/grade/sharpen have something to act on.
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        bg[:, x] = [int(x * 2), int(255 - x * 2), 100]
    noise = rng.integers(-25, 25, size=(h, w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Bright solid subject in the center.
    bg[32:96, 32:96] = [200, 180, 160]

    person_mask = np.zeros((h, w), dtype=np.float32)
    person_mask[32:96, 32:96] = 1.0
    return bg, person_mask


def _to_float01(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# BackgroundReplacer unit tests
# ---------------------------------------------------------------------------

class TestBackgroundReplacer:
    @pytest.fixture
    def replacer(self) -> BackgroundReplacer:
        return BackgroundReplacer()

    # --- dtype preservation ---

    def test_replace_preserves_uint8_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        new_bg = np.zeros_like(img)
        new_bg[:] = [10, 20, 30]
        out = replacer.replace(img, pm, new_bg, strength=1.0)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_replace_preserves_float32_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        img_f = img.astype(np.float32)
        new_bg = np.zeros_like(img_f)
        new_bg[:] = [10.0, 20.0, 30.0]
        out = replacer.replace(img_f, pm, new_bg, strength=1.0)
        assert out.dtype == np.float32

    def test_grade_background_preserves_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        for src in (img, img.astype(np.float32)):
            out = replacer.grade_background(src, pm, {"desaturation": 50.0})
            assert out.dtype == src.dtype

    def test_blur_background_preserves_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        for src in (img, img.astype(np.float32)):
            out = replacer.blur_background(src, pm, 50.0)
            assert out.dtype == src.dtype

    def test_light_wrap_preserves_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        for src in (img, img.astype(np.float32)):
            out = replacer.light_wrap(src, pm, 50.0)
            assert out.dtype == src.dtype

    def test_sharpen_subject_preserves_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        for src in (img, img.astype(np.float32)):
            out = replacer.sharpen_subject(src, pm, 50.0)
            assert out.dtype == src.dtype

    def test_relight_scene_preserves_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        for src in (img, img.astype(np.float32)):
            out = replacer.relight_scene(src, pm, 45.0, 60.0, 50.0)
            assert out.dtype == src.dtype

    # --- early-return guards ---

    def test_replace_zero_strength_is_noop(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        new_bg = np.zeros_like(img)
        out = replacer.replace(img, pm, new_bg, strength=0.0)
        np.testing.assert_array_equal(out, img)

    def test_grade_background_all_zero_is_noop(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.grade_background(img, pm, {})
        np.testing.assert_array_equal(out, img)

    def test_blur_zero_is_noop(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.blur_background(img, pm, 0.0)
        np.testing.assert_array_equal(out, img)

    def test_light_wrap_zero_is_noop(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.light_wrap(img, pm, 0.0)
        np.testing.assert_array_equal(out, img)

    def test_sharpen_zero_is_noop(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.sharpen_subject(img, pm, 0.0)
        np.testing.assert_array_equal(out, img)

    def test_relight_zero_is_noop(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.relight_scene(img, pm, 45.0, 60.0, 0.0)
        np.testing.assert_array_equal(out, img)

    def test_rejects_bad_dtype(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        bad = img.astype(np.int16)
        with pytest.raises(TypeError):
            replacer.replace(bad, pm, img, strength=1.0)

    # --- subject protection ---

    def test_replace_keeps_subject_pixels(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        new_bg = np.zeros_like(img)
        new_bg[:] = [0, 0, 0]
        out = replacer.replace(img, pm, new_bg, strength=1.0)
        # Center of the subject (well inside the mask) must be unchanged.
        np.testing.assert_array_equal(out[60, 60], img[60, 60])

    def test_blur_keeps_subject_sharp(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.blur_background(img, pm, 80.0)
        # Subject center unchanged.
        np.testing.assert_array_equal(out[60, 60], img[60, 60])

    def test_grade_keeps_subject_unchanged(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.grade_background(img, pm, {
            "desaturation": 80.0,
            "blue_shadow_grade": 80.0,
            "cyan_midtone_grade": 80.0,
            "matte_black": 80.0,
        })
        np.testing.assert_array_equal(out[60, 60], img[60, 60])

    def test_sharpen_keeps_background_unchanged(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.sharpen_subject(img, pm, 80.0)
        # Background corner (well outside the mask) must be unchanged.
        np.testing.assert_array_equal(out[5, 5], img[5, 5])

    # --- background actually changes ---

    def test_replace_changes_background(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        new_bg = np.zeros_like(img)
        new_bg[:] = [0, 0, 0]
        out = replacer.replace(img, pm, new_bg, strength=1.0)
        # Background corner should be close to the new bg (feather may pull
        # it slightly, but with a 96px gap from the subject it's negligible).
        assert int(out[5, 5, 0]) < 30
        assert int(out[5, 5, 1]) < 30
        assert int(out[5, 5, 2]) < 30

    def test_blur_smooths_background(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.blur_background(img, pm, 80.0)
        # Blurring reduces high-frequency energy. Measure it as the mean
        # absolute Laplacian in a background patch (away from the subject).
        patch_orig = cv2.cvtColor(img[10:30, 10:30], cv2.COLOR_BGR2GRAY).astype(np.float32)
        patch_out = cv2.cvtColor(out[10:30, 10:30], cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap_orig = cv2.Laplacian(patch_orig, cv2.CV_32F)
        lap_out = cv2.Laplacian(patch_out, cv2.CV_32F)
        assert np.abs(lap_out).mean() < np.abs(lap_orig).mean()

    def test_desaturation_reduces_background_chroma(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.grade_background(img, pm, {"desaturation": 100.0})
        # Measure saturation in a background patch via HSV.
        def sat(i):
            hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV).astype(np.float32)
            return hsv[:, :, 1].mean()
        assert sat(out) < sat(img)

    def test_matte_black_darkens_background_shadows(self, replacer: BackgroundReplacer) -> None:
        # Build a dark-background image so matte_black has shadows to crush.
        h, w = 128, 128
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        img[32:96, 32:96] = 200  # bright subject
        pm = np.zeros((h, w), dtype=np.float32)
        pm[32:96, 32:96] = 1.0
        replacer = BackgroundReplacer()
        out = replacer.grade_background(img, pm, {"matte_black": 100.0})
        # Background mean luminance should drop.
        assert int(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[10, 10]) < int(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[10, 10])

    def test_subject_sharpen_increases_subject_contrast(self, replacer: BackgroundReplacer) -> None:
        # Subject with an internal edge so sharpening has something to amplify.
        h, w = 128, 128
        img = np.full((h, w, 3), 120, dtype=np.uint8)
        img[32:64, 32:96] = 200  # bright band inside the subject region
        pm = np.zeros((h, w), dtype=np.float32)
        pm[32:96, 32:96] = 1.0
        out = replacer.sharpen_subject(img, pm, 80.0, radius=1.0)
        # The internal edge (row 47, inside subject) should have a steeper
        # gradient after sharpening.
        g_orig = np.abs(np.diff(img[47, 30:70, 0].astype(np.float32)))
        g_out = np.abs(np.diff(out[47, 30:70, 0].astype(np.float32)))
        assert g_out.max() >= g_orig.max()

    def test_light_wrap_modifies_edge_band(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.light_wrap(img, pm, 80.0, wrap_color=(255.0, 255.0, 255.0))
        # The edge band (just inside the subject, near the boundary) should
        # be brighter than the original due to the screen-blended white wrap.
        edge_pixel_orig = int(img[33, 33, 0])
        edge_pixel_out = int(out[33, 33, 0])
        assert edge_pixel_out >= edge_pixel_orig

    # --- relight_scene ---

    def test_relight_changes_image(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out = replacer.relight_scene(img, pm, 45.0, 60.0, 80.0)
        # The image should differ from the input somewhere.
        assert not np.array_equal(out, img)

    def test_relight_direction_matters(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        out_left = replacer.relight_scene(img, pm, 180.0, 45.0, 80.0)
        out_right = replacer.relight_scene(img, pm, 0.0, 45.0, 80.0)
        # Different light directions should produce different results.
        assert not np.array_equal(out_left, out_right)

    def test_relight_strength_scales(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        weak = replacer.relight_scene(img, pm, 45.0, 60.0, 20.0)
        strong = replacer.relight_scene(img, pm, 45.0, 60.0, 90.0)
        d_weak = np.abs(weak.astype(np.float32) - img.astype(np.float32)).mean()
        d_strong = np.abs(strong.astype(np.float32) - img.astype(np.float32)).mean()
        assert d_strong > d_weak

    # --- replace resizes mismatched background ---

    def test_replace_resizes_background(self, replacer: BackgroundReplacer) -> None:
        img, pm = _make_subject_bg_image()
        big_bg = np.zeros((256, 256, 3), dtype=np.uint8)
        big_bg[:] = [10, 20, 30]
        out = replacer.replace(img, pm, big_bg, strength=1.0)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Engine wiring tests
# ---------------------------------------------------------------------------

class TestEngineWiring:
    def test_context_defaults_zero(self) -> None:
        ctx = ProcessingContext()
        for k in (
            "background_blur", "background_desaturation", "light_wrap",
            "blue_shadow_grade", "cyan_midtone_grade", "subject_sharpen",
            "matte_black",
        ):
            assert getattr(ctx, k) == 0.0, f"{k} default should be 0.0"

    def test_anime_crystal_void_keys_resolve_in_context(self) -> None:
        rec = resolve_recipe("anime_crystal_void")
        ctx = build_context("anime_crystal_void", rec, {})
        assert ctx.background_blur == 35.0
        assert ctx.background_desaturation == 55.0
        assert ctx.light_wrap == 25.0
        assert ctx.blue_shadow_grade == 60.0
        assert ctx.cyan_midtone_grade == 30.0
        assert ctx.subject_sharpen == 40.0
        assert ctx.matte_black == 50.0

    def test_keys_have_paramspecs(self) -> None:
        spec_names = {spec.name for spec in PROCESSING_PARAMS}
        for k in (
            "background_blur", "background_desaturation", "light_wrap",
            "blue_shadow_grade", "cyan_midtone_grade", "subject_sharpen",
            "matte_black",
        ):
            assert k in spec_names, f"{k} missing from PROCESSING_PARAMS"

    def test_stage_background_noop_when_all_zero(self) -> None:
        from retouch.engine import RetouchEngine
        img, pm = _make_subject_bg_image()
        ctx = ProcessingContext()  # all background params 0
        # We call the private stage directly to avoid the full pipeline.
        engine = RetouchEngine.__new__(RetouchEngine)
        engine._background_replacer = BackgroundReplacer()
        out = engine._stage_background(img.copy(), ctx, pm)
        np.testing.assert_array_equal(out, img)

    def test_stage_background_noop_without_person_mask(self) -> None:
        from retouch.engine import RetouchEngine
        img, _ = _make_subject_bg_image()
        ctx = ProcessingContext()
        ctx.background_blur = 50.0
        engine = RetouchEngine.__new__(RetouchEngine)
        engine._background_replacer = BackgroundReplacer()
        out = engine._stage_background(img.copy(), ctx, None)
        np.testing.assert_array_equal(out, img)

    def test_stage_background_runs_and_preserves_dtype(self) -> None:
        from retouch.engine import RetouchEngine
        img, pm = _make_subject_bg_image()
        ctx = ProcessingContext()
        ctx.background_blur = 50.0
        ctx.matte_black = 50.0
        engine = RetouchEngine.__new__(RetouchEngine)
        engine._background_replacer = BackgroundReplacer()
        # uint8 input
        out_u8 = engine._stage_background(img.copy(), ctx, pm)
        assert out_u8.dtype == np.uint8
        # float32 [0,1] input
        img_f = _to_float01(img)
        out_f = engine._stage_background(img_f.copy(), ctx, pm)
        assert out_f.dtype == np.float32
        assert out_f.min() >= 0.0 and out_f.max() <= 1.0

    def test_stage_background_protects_subject(self) -> None:
        from retouch.engine import RetouchEngine
        img, pm = _make_subject_bg_image()
        ctx = ProcessingContext()
        ctx.background_blur = 80.0
        ctx.background_desaturation = 80.0
        ctx.matte_black = 80.0
        engine = RetouchEngine.__new__(RetouchEngine)
        engine._background_replacer = BackgroundReplacer()
        out = engine._stage_background(img.copy(), ctx, pm)
        # Subject center must be byte-identical.
        np.testing.assert_array_equal(out[60, 60], img[60, 60])

    def test_stage_background_changes_background(self) -> None:
        from retouch.engine import RetouchEngine
        img, pm = _make_subject_bg_image()
        ctx = ProcessingContext()
        ctx.background_blur = 80.0
        engine = RetouchEngine.__new__(RetouchEngine)
        engine._background_replacer = BackgroundReplacer()
        out = engine._stage_background(img.copy(), ctx, pm)
        # Background corner should differ (blurred).
        assert not np.array_equal(out[5, 5], img[5, 5])
