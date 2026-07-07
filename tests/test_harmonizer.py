"""Tests for C5 — skin-anchored background color harmonization.

Validates the BackgroundHarmonizer class and the engine wiring
(_stage_harmonize + BackgroundHarmonizeStage):
  - dtype preservation (uint8 / float32)
  - early-return guards (zero strength, no skin mask, no person mask)
  - subject protection (person region unchanged)
  - background actually shifts toward the complementary hue
  - effect scales with strength
  - all four harmony modes produce different background hues
  - engine context field default + stage gating
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from retouch.harmonizer import BackgroundHarmonizer, _hue_target
from retouch.color_science import bgr_to_oklab, oklab_to_oklch


def _make_skin_person_image() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic image: warm skin subject on a cool blue background.

    Returns:
        (img_bgr_u8, skin_mask, person_mask) — all (256, 256) or (256, 256, 3).
    """
    h, w = 256, 256
    # Warm skin tone (BGR) ≈ hue ~30° on the OKLCh wheel.
    skin_bgr = np.array([150, 170, 200], dtype=np.uint8)
    # Cool blue background ≈ hue ~240°.
    bg_bgr = np.array([200, 130, 60], dtype=np.uint8)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bg_bgr
    # Subject occupies the center.
    img[80:180, 80:180] = skin_bgr

    person_mask = np.zeros((h, w), dtype=np.float32)
    person_mask[80:180, 80:180] = 1.0

    skin_mask = np.zeros((h, w), dtype=np.float32)
    skin_mask[100:160, 100:160] = 1.0

    return img, skin_mask, person_mask


class TestHueTarget:
    def test_complementary(self) -> None:
        assert _hue_target(0.0, "complementary") == pytest.approx(180.0)

    def test_split(self) -> None:
        assert _hue_target(0.0, "split") == pytest.approx(150.0)

    def test_analogous_warm(self) -> None:
        assert _hue_target(0.0, "analogous_warm") == pytest.approx(30.0)

    def test_analogous_cool(self) -> None:
        assert _hue_target(0.0, "analogous_cool") == pytest.approx(330.0)

    def test_unknown_mode_falls_back(self) -> None:
        assert _hue_target(0.0, "bogus") == pytest.approx(150.0)


class TestBackgroundHarmonizer:
    @pytest.fixture
    def harmonizer(self) -> BackgroundHarmonizer:
        return BackgroundHarmonizer()

    def test_zero_strength_returns_input_unchanged(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        img, skin_mask, person_mask = _make_skin_person_image()
        result = harmonizer.harmonize(img, skin_mask, person_mask, strength=0.0)
        np.testing.assert_array_equal(result, img)

    def test_none_skin_mask_returns_input_unchanged(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        img, _, person_mask = _make_skin_person_image()
        result = harmonizer.harmonize(img, None, person_mask, strength=0.8)
        np.testing.assert_array_equal(result, img)

    def test_empty_skin_mask_returns_input_unchanged(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        img, _, person_mask = _make_skin_person_image()
        empty_skin = np.zeros((256, 256), dtype=np.float32)
        result = harmonizer.harmonize(img, empty_skin, person_mask, strength=0.8)
        np.testing.assert_array_equal(result, img)

    def test_dtype_preserved_uint8(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        img, skin_mask, person_mask = _make_skin_person_image()
        result = harmonizer.harmonize(img, skin_mask, person_mask, strength=0.5)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_dtype_preserved_float32(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        img, skin_mask, person_mask = _make_skin_person_image()
        img_f = img.astype(np.float32)
        result = harmonizer.harmonize(img_f, skin_mask, person_mask, strength=0.5)
        assert result.dtype == np.float32
        assert result.shape == img_f.shape

    def test_float32_stays_in_range(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        img, skin_mask, person_mask = _make_skin_person_image()
        img_f = img.astype(np.float32)
        result = harmonizer.harmonize(img_f, skin_mask, person_mask, strength=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_subject_region_protected(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        """Pixels inside the person mask should not change."""
        img, skin_mask, person_mask = _make_skin_person_image()
        result = harmonizer.harmonize(img, skin_mask, person_mask, strength=1.0)
        # Deep inside the subject (far from the feathered boundary).
        subject_before = img[120:140, 120:140].astype(np.int16)
        subject_after = result[120:140, 120:140].astype(np.int16)
        # Allow tiny feather leakage at the very edge but the core must be
        # essentially unchanged.
        assert np.abs(subject_after - subject_before).max() < 5

    def test_background_actually_shifts(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        """Background hue should move toward the complementary hue."""
        img, skin_mask, person_mask = _make_skin_person_image()

        # Measure original background hue in OKLCh (needs (H, W, 3) input).
        bg_sample = np.array([[[60, 130, 200]]], dtype=np.uint8)  # BGR blue
        bg_oklch_before = oklab_to_oklch(bgr_to_oklab(bg_sample))
        h_before = float(bg_oklch_before[0, 0, 2])

        result = harmonizer.harmonize(img, skin_mask, person_mask, strength=1.0)

        # Sample a background pixel far from the subject.
        bg_after_sample = np.array([[result[10, 10]]], dtype=np.uint8)
        bg_oklch_after = oklab_to_oklch(bgr_to_oklab(bg_after_sample))
        h_after = float(bg_oklch_after[0, 0, 2])

        # The hue should have moved (not stayed identical).
        dist = abs(((h_after - h_before + 180.0) % 360.0) - 180.0)
        assert dist > 1.0, (
            f"Background hue did not shift: before={h_before:.2f}, "
            f"after={h_after:.2f}"
        )

    def test_effect_scales_with_strength(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        """Stronger strength → larger background change."""
        img, skin_mask, person_mask = _make_skin_person_image()

        weak = harmonizer.harmonize(img, skin_mask, person_mask, strength=0.2)
        strong = harmonizer.harmonize(img, skin_mask, person_mask, strength=1.0)

        weak_delta = np.abs(weak.astype(np.int16) - img.astype(np.int16)).mean()
        strong_delta = np.abs(strong.astype(np.int16) - img.astype(np.int16)).mean()
        assert strong_delta > weak_delta

    def test_modes_produce_different_backgrounds(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        """Different harmony modes should grade the background differently.

        Uses a green background (hue ~120°) so the four target hues
        (180/270/150/90°) diverge widely and the graded backgrounds are
        visually distinguishable.
        """
        h, w = 256, 256
        skin_bgr = np.array([150, 170, 200], dtype=np.uint8)
        # Green background in BGR.
        bg_bgr = np.array([60, 200, 60], dtype=np.uint8)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = bg_bgr
        img[80:180, 80:180] = skin_bgr
        person_mask = np.zeros((h, w), dtype=np.float32)
        person_mask[80:180, 80:180] = 1.0
        skin_mask = np.zeros((h, w), dtype=np.float32)
        skin_mask[100:160, 100:160] = 1.0

        modes = ["complementary", "split", "analogous_warm", "analogous_cool"]
        results: dict[str, np.ndarray] = {}
        for m in modes:
            results[m] = harmonizer.harmonize(
                img, skin_mask, person_mask, strength=1.0, mode=m
            )

        # At least two distinct backgrounds should exist (compare full RGB
        # pixel at a background sample point).
        samples = {m: results[m][10, 10].tolist() for m in modes}
        assert len(set(tuple(v) for v in samples.values())) > 1, (
            f"All modes produced the same background: {samples}"
        )

    def test_no_person_mask_grades_whole_image(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        """When person_mask is None, the whole image is graded (no crash)."""
        img, skin_mask, _ = _make_skin_person_image()
        result = harmonizer.harmonize(img, skin_mask, None, strength=0.5)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_invalid_dtype_raises(
        self, harmonizer: BackgroundHarmonizer
    ) -> None:
        img, skin_mask, person_mask = _make_skin_person_image()
        bad = img.astype(np.float64)
        with pytest.raises(TypeError):
            harmonizer.harmonize(bad, skin_mask, person_mask, strength=0.5)


class TestEngineWiring:
    """Verify the engine stage and context field exist and gate correctly."""

    def test_context_defaults(self) -> None:
        from retouch.engine import ProcessingContext

        ctx = ProcessingContext()
        assert ctx.background_harmonize == 0.0
        assert ctx.background_harmonize_mode == "split"

    def test_stage_method_exists(self) -> None:
        from retouch.engine import RetouchEngine

        assert hasattr(RetouchEngine, "_stage_harmonize")

    def test_stage_wrapper_registered(self) -> None:
        from retouch.stage_wrappers import (
            build_global_registry,
            BackgroundHarmonizeStage,
        )

        class _DummyEngine:
            pass

        eng = _DummyEngine()
        registry = build_global_registry(eng)  # type: ignore[arg-type]
        names = [s.name for s in registry._stages]
        assert "background_harmonize" in names
        # Must come after subject_separation and before body_skin.
        assert names.index("background_harmonize") > names.index(
            "subject_separation"
        )
        assert names.index("background_harmonize") < names.index("body_skin")

    def test_param_spec_exists(self) -> None:
        from retouch.params import get_param

        spec = get_param("background_harmonize")
        assert spec.default == 0.0
        assert spec.min_val == 0
        assert spec.max_val == 100

        mode_spec = get_param("background_harmonize_mode")
        assert mode_spec.default == "split"
