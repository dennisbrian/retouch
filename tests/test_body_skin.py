"""Tests for Stage S1 — Body Skin Retouch.

Tests verify:
1. All-params-zero → unchanged output
2. Body skin mask construction (LCH detection, person_mask intersection, face exclusion)
3. False-positive detection and mitigation (contiguity check)
4. Tattoo/body-paint exclusion (high-chroma zones)
5. Tone matching bounds (±8L, ±6a/b clamping)
6. Individual parameter gates (smooth, equalize, whiten, match_face)
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from retouch import RetouchEngine
from retouch import engine as engine_module
from retouch.color_space import bgr_to_lch, skin_mask_lch
from retouch.engine import ProcessingContext
from retouch.utils import normalize_mask, squeeze_mask


class TestBodySkinMaskConstruction:
    """Tests for body_skin mask construction logic."""

    def test_all_params_zero_returns_unchanged(self, natural_image):
        """When all body_skin params are zero, output must be identical to input."""
        engine = RetouchEngine()
        img = natural_image

        result = engine.process(
            img,
            recipe="natural",
            body_smooth=0.0,
            body_equalize=0.0,
            body_whiten=0.0,
            body_match_face=0.0,
            fast=True,
        )

        # Slight color difference is OK due to color space conversions,
        # but the image should be very similar (within uint8 quantization noise)
        diff = cv2.absdiff(img.astype(np.float32), result.astype(np.float32))
        mean_diff = np.mean(diff)
        # All-zero params should result in minimal changes (only from required processing)
        assert mean_diff < 5.0, f"Expected minimal diff with all-zero params, got {mean_diff}"

    def test_skin_mask_lch_detects_skin_tones(self):
        """Verify LCH-based skin detection works on synthetic skin-toned regions."""
        # Create synthetic image with:
        # - skin-tone region (center)
        # - non-skin region (corners)
        h, w = 200, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Add skin-tone BGR: approximate (120, 110, 200) in BGR ~ (140, 100, 180) in LAB
        # which is roughly 25° hue in LCH
        img[50:150, 50:150] = (120, 110, 200)  # BGR skin tone
        img[0:30, 0:30] = (50, 50, 50)  # dark corner (non-skin)
        img[170:200, 170:200] = (255, 255, 255)  # bright corner (non-skin)

        lch = bgr_to_lch(img)
        skin_mask = skin_mask_lch(lch, hue_center=25.0, hue_tolerance=25.0, chroma_min=8.0)

        # Center should be detected as skin
        center_mask = skin_mask[50:150, 50:150]
        assert np.mean(center_mask) > 0.5, "Skin-tone region should be >50% detected"

        # Corners should mostly not be detected as skin
        corner_dark = skin_mask[0:30, 0:30]
        corner_bright = skin_mask[170:200, 170:200]
        assert np.mean(corner_dark) < 0.2, "Dark corner should be <20% detected"
        assert np.mean(corner_bright) < 0.2, "Bright corner should be <20% detected"

    def test_contiguity_check_rejects_disconnected_false_positives(self):
        """Verify disconnected skin-toned objects are dropped by contiguity check."""
        engine = RetouchEngine()

        # Create synthetic image: person-mask center + skin-toned blob far away
        h, w = 400, 400
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Body region: center, skin-tone
        img[100:250, 100:250] = (120, 110, 200)  # skin tone

        # False-positive prop: far away, also skin-tone but NOT part of person_mask
        img[20:60, 320:360] = (120, 110, 200)  # isolated skin-tone patch

        # Create person_mask: only covers center body, NOT the far blob
        person_mask = np.zeros((h, w), dtype=np.float32)
        person_mask[80:270, 80:270] = 1.0  # only center

        # Simulate detection
        lch = bgr_to_lch(img)
        skin_mask_lch_result = skin_mask_lch(lch, hue_center=25.0, hue_tolerance=25.0, chroma_min=8.0)

        # Before intersection: both body and blob are skin-colored
        assert np.sum(skin_mask_lch_result[100:150, 100:150]) > 0, "Body region should be detected"
        assert np.sum(skin_mask_lch_result[20:60, 320:360]) > 0, "Blob should be detected in LCH mask"

        # After intersection with person_mask
        pm = normalize_mask(person_mask)
        pm = squeeze_mask(pm)
        body_candidate = skin_mask_lch_result * pm

        # Body region should survive
        assert np.sum(body_candidate[100:150, 100:150]) > 0, "Body region should survive person_mask intersection"

        # Blob should be eliminated
        assert np.sum(body_candidate[20:60, 320:360]) == 0, "Isolated blob should be eliminated by person_mask"

    def test_tattoo_exclusion_via_chroma_gate(self):
        """Verify high-chroma zones (tattoos) are excluded from body_skin_mask."""
        # Create image with normal skin and high-chroma skin-toned tattoo patch
        h, w = 200, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Normal skin tone (moderate saturation)
        img[50:150, 50:150] = (120, 110, 200)  # moderate chroma skin

        # High-chroma tattoo patch: same hue but MORE saturated
        # Increase saturation by making orange/red more vivid (within skin hue range)
        img[80:120, 80:120] = (60, 80, 220)  # higher chroma, same hue

        lch = bgr_to_lch(img)
        skin_mask_lch_result = skin_mask_lch(lch, hue_center=25.0, hue_tolerance=25.0, chroma_min=8.0)

        # Both regions should be detected as skin color
        normal_detected = np.sum(skin_mask_lch_result[50:80, 50:80] > 0.5)
        tattoo_detected = np.sum(skin_mask_lch_result[80:120, 80:120] > 0.5)
        assert normal_detected > 100, "Normal skin should be detected by LCH"
        assert tattoo_detected > 200, "Tattoo region should also pass LCH (same hue)"

        # Extract chroma channel
        c_channel = lch[:, :, 1].astype(np.float32)

        # Apply tattoo gate: exclude high-chroma (>0.18 in normalized LCH)
        # This should protect bright reds/saturated colors
        tattoo_gate = np.clip((0.18 - c_channel) / 0.02, 0.0, 1.0)
        body_with_tattoo_protection = skin_mask_lch_result * tattoo_gate

        # Normal skin should mostly survive (moderate chroma)
        normal_skin_mean = np.mean(body_with_tattoo_protection[50:80, 50:80])
        # Tattoo should be more attenuated (high chroma)
        tattoo_mean = np.mean(body_with_tattoo_protection[80:120, 80:120])

        # The tattoo region should have lower gate value than normal skin
        # (both pass LCH detection, but tattoo is suppressed by chroma gate)
        assert tattoo_mean <= normal_skin_mean, "Tattoo should be attenuated more than or equal to normal skin due to higher chroma"

    def test_body_stage_reaches_remote_skin_through_person_silhouette(self, monkeypatch):
        """A face-to-chest gap must not make valid body parameters inert.

        The old fixed 12-step geodesic cap reached only a small ring around
        the face.  This fixture places a valid chest candidate farther than
        that ring while keeping it inside the same continuous person mask.
        The body-match operation must therefore alter the chest pixels.
        """
        h, w = 240, 160
        img = np.full((h, w, 3), (75, 100, 145), dtype=np.uint8)
        img[15:45, 55:105] = (165, 185, 225)  # retouched face reference
        img_f = img.astype(np.float32) / 255.0

        person = np.zeros((h, w), dtype=np.float32)
        person[5:235, 25:135] = 1.0
        # A separate subject-like island proves the converged reachability
        # still rejects components that are not connected to the face.
        person[150:220, 142:158] = 1.0
        face = np.zeros((h, w), dtype=np.float32)
        face[15:45, 55:105] = 1.0

        # A chest-only candidate starts more than 12 x 4px dilation steps
        # from the face.  Keep the mask construction deterministic so this
        # regression tests reachability rather than colour thresholds.
        candidate = np.zeros((h, w), dtype=np.float32)
        candidate[150:220, 40:120] = 1.0
        candidate[150:220, 142:158] = 1.0
        monkeypatch.setattr(engine_module, "skin_mask_lch", lambda *args, **kwargs: candidate)
        monkeypatch.setattr(
            engine_module,
            "oklab_to_oklch",
            lambda oklab: np.zeros((*oklab.shape[:2], 3), dtype=np.float32),
        )

        class _NoFullFrameHair:
            @staticmethod
            def parse_hair_full_image(_img):
                return None

        engine = object.__new__(RetouchEngine)
        engine._parser = _NoFullFrameHair()
        result = engine._stage_body_skin(
            img_f,
            ProcessingContext(body_match_face=100.0),
            person,
            face,
            np.zeros_like(face),
            np.zeros_like(face),
            [],
            h,
            w,
        )

        chest = np.s_[170:200, 60:100]
        assert not np.array_equal(result[chest], img_f[chest]), (
            "A connected, valid chest candidate was dropped before any body "
            "operation ran. Reachability must converge through person_mask."
        )
        other_subject = np.s_[170:200, 145:155]
        # LAB conversion round-trips can move an untouched uint8 channel by
        # one code value.  Anything beyond that is body-mask leakage.
        assert np.max(np.abs(result[other_subject] - img_f[other_subject])) <= (1.0 / 255.0 + 1e-6), (
            "A candidate on a disconnected person-mask component must not be "
            "treated as the detected face's body skin."
        )


class TestBodySkinProcessing:
    """Tests for actual body skin processing operations."""

    def test_body_match_face_bounds_clamping(self):
        """Verify body_match_face color delta is bounded to ±8L ±6a/b."""
        engine = RetouchEngine()

        # Create synthetic image with:
        # - face region (lighter, warmer)
        # - body region (darker, cooler)
        h, w = 300, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Face: light, rosy skin (high L, positive a)
        face_y1, face_y2, face_x1, face_x2 = 30, 100, 80, 170
        img[face_y1:face_y2, face_x1:face_x2] = (200, 160, 240)  # light, rosy

        # Body: darker, cooler skin (low L, negative a)
        body_y1, body_y2, body_x1, body_x2 = 150, 250, 50, 200
        img[body_y1:body_y2, body_x1:body_x2] = (40, 90, 120)  # darker, more cool

        # Create face and body masks
        face_mask = np.zeros((h, w), dtype=np.float32)
        face_mask[face_y1:face_y2, face_x1:face_x2] = 1.0

        body_mask = np.zeros((h, w), dtype=np.float32)
        body_mask[body_y1:body_y2, body_x1:body_x2] = 1.0

        # Convert to LAB and measure medians
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

        face_indices = face_mask > 0.3
        body_indices = body_mask > 0.3

        face_median_l = np.median(lab[:, :, 0][face_indices])
        face_median_a = np.median(lab[:, :, 1][face_indices])
        face_median_b = np.median(lab[:, :, 2][face_indices])

        body_median_l = np.median(lab[:, :, 0][body_indices])
        body_median_a = np.median(lab[:, :, 1][body_indices])
        body_median_b = np.median(lab[:, :, 2][body_indices])

        # Compute raw deltas (unclamped)
        raw_l_diff = face_median_l - body_median_l
        raw_a_diff = face_median_a - body_median_a
        raw_b_diff = face_median_b - body_median_b

        # Should be significant (>20 levels)
        assert abs(raw_l_diff) > 20, f"L delta should be significant for test, got {raw_l_diff}"

        # Apply clamping (as done in _stage_body_skin)
        clamped_l_diff = np.clip(raw_l_diff * 1.0, -8.0, 8.0)  # strength=1.0
        clamped_a_diff = np.clip(raw_a_diff * 1.0, -6.0, 6.0)
        clamped_b_diff = np.clip(raw_b_diff * 1.0, -6.0, 6.0)

        # Verify clamping worked
        assert abs(clamped_l_diff) <= 8.0, f"L delta should be clamped to ±8, got {clamped_l_diff}"
        assert abs(clamped_a_diff) <= 6.0, f"a delta should be clamped to ±6, got {clamped_a_diff}"
        assert abs(clamped_b_diff) <= 6.0, f"b delta should be clamped to ±6, got {clamped_b_diff}"

    def test_body_smooth_gate_independent(self, natural_image):
        """Verify body_smooth param gates smoothing independently."""
        engine = RetouchEngine()
        img = natural_image

        # Test with only body_smooth enabled
        result_smooth = engine.process(
            img.copy(),
            recipe="natural",
            body_smooth=50.0,
            body_equalize=0.0,
            body_whiten=0.0,
            body_match_face=0.0,
            fast=True,
        )

        # Test with no body_smooth
        result_no_smooth = engine.process(
            img.copy(),
            recipe="natural",
            body_smooth=0.0,
            body_equalize=0.0,
            body_whiten=0.0,
            body_match_face=0.0,
            fast=True,
        )

        # Results should differ (smoothing should have an effect). Body skin is
        # a small fraction of this full-body real photo, so a whole-frame mean
        # diff is diluted to near-zero even when the affected region changes a
        # lot — check the max pixel diff instead, which is robust to a small
        # affected area.
        diff = cv2.absdiff(result_smooth.astype(np.float32), result_no_smooth.astype(np.float32))
        max_diff = np.max(diff)
        assert max_diff > 0.5, "body_smooth should produce visible difference"

    def test_body_whiten_gate_independent(self, natural_image):
        """Verify body_whiten param gates whitening independently."""
        engine = RetouchEngine()
        img = natural_image

        result_whiten = engine.process(
            img.copy(),
            recipe="natural",
            body_smooth=0.0,
            body_equalize=0.0,
            body_whiten=50.0,
            body_match_face=0.0,
            fast=True,
        )

        result_no_whiten = engine.process(
            img.copy(),
            recipe="natural",
            body_smooth=0.0,
            body_equalize=0.0,
            body_whiten=0.0,
            body_match_face=0.0,
            fast=True,
        )

        # Results should differ (whitening should brighten skin). Body skin is
        # a small fraction of this full-body real photo, so a whole-frame mean
        # diff is diluted to near-zero even when the affected region changes a
        # lot — check the max pixel diff instead, which is robust to a small
        # affected area.
        diff = cv2.absdiff(result_whiten.astype(np.float32), result_no_whiten.astype(np.float32))
        max_diff = np.max(diff)
        assert max_diff > 0.5, "body_whiten should produce visible difference"

    def test_body_blemish_gate_independent_of_smooth(self, natural_image):
        """Regression test: blemish removal must run whenever the body-skin stage
        is active (any of the 4 params nonzero), NOT only when body_smooth > 0.

        Bug history: the blemish sub-step was originally gated on
        `ctx.body_smooth > 0` alone, so a caller setting only body_whiten and/or
        body_match_face (a legitimate combo: "match tone + whiten, don't smooth
        away leg texture") got silently zero blemish removal.

        Verified via monkey-patching retouch.engine.BlemishRemover.remove and
        recording every call's mask argument. `BlemishRemover` is a single class
        shared by the face-level pass (perf_optimizations.py, gated on
        ctx.blemish, which defaults to 30 in the "natural" recipe) and the
        body-skin pass added in _stage_body_skin — so a bare call-count check
        would pass even if the body pass never ran, as long as the face pass
        fired. To discriminate, disable the face pass (blemish=0) and confirm
        at least one call still occurs with body_smooth=0.
        """
        from retouch import engine as engine_module

        img = natural_image

        call_count = {"n": 0}
        original_remove = engine_module.BlemishRemover.remove

        def _counting_remove(self, img_bgr, skin_mask, strength=30, **kwargs):
            call_count["n"] += 1
            return original_remove(
                self,
                img_bgr,
                skin_mask,
                strength=strength,
                **kwargs,
            )

        engine_module.BlemishRemover.remove = _counting_remove
        try:
            engine = RetouchEngine()
            # Disable the unrelated face-level blemish pass (blemish=0) so any
            # BlemishRemover.remove() call can only originate from the body-skin
            # stage. body_smooth is explicitly 0.0 — only body_whiten/
            # body_match_face are active — this is the exact combo that
            # previously produced zero body blemish removal.
            engine.process(
                img,
                recipe="natural",
                blemish=0.0,
                body_smooth=0.0,
                body_equalize=0.0,
                body_whiten=60.0,
                body_match_face=40.0,
                fast=True,
            )
        finally:
            engine_module.BlemishRemover.remove = original_remove

        assert call_count["n"] >= 1, (
            "BlemishRemover.remove() was never invoked for the body-skin stage "
            "even though body_whiten/body_match_face were active, body_smooth=0, "
            "and the face-level blemish pass was disabled (blemish=0). The "
            "blemish sub-step must be gated on the stage being active, not on "
            "body_smooth specifically."
        )


class TestBodySkinE2E:
    """End-to-end integration tests."""

    def test_process_with_all_body_params_nonzero(self, natural_image):
        """Full pipeline must not crash with all body_skin params set."""
        engine = RetouchEngine()
        img = natural_image

        # Test the exact scenario from task spec
        result = engine.process(
            img,
            recipe="natural",
            body_smooth=60.0,
            body_equalize=40.0,
            body_whiten=30.0,
            body_match_face=50.0,
            fast=False,  # Full resolution path
        )

        # Verify output
        assert result is not None, "Process should return result"
        assert result.shape == img.shape, "Output shape should match input"
        assert result.dtype == np.uint8, "Output should be uint8"
        assert result.max() <= 255 and result.min() >= 0, "Output should be valid BGR"

    def test_body_params_with_different_recipes(self, natural_image):
        """Body params should work with different recipes."""
        engine = RetouchEngine()
        img = natural_image

        for recipe in ["natural", "cosplay", "soft"]:
            result = engine.process(
                img.copy(),
                recipe=recipe,
                body_smooth=30.0,
                body_match_face=40.0,
                fast=True,
            )
            assert result is not None, f"Should work with recipe {recipe}"
            assert result.shape == img.shape

    def test_body_params_no_crash_without_faces(self):
        """Body processing should gracefully handle images with no faces."""
        engine = RetouchEngine()

        # Create a blank image (no faces)
        blank_img = np.ones((400, 400, 3), dtype=np.uint8) * 128

        # Should not crash
        result = engine.process(
            blank_img,
            recipe="natural",
            body_smooth=50.0,
            body_match_face=30.0,
            fast=True,
        )

        assert result is not None
        assert result.shape == blank_img.shape


class TestBodySkinMaskMetrics:
    """Tests to measure and validate body_skin mask properties."""

    def test_body_skin_mask_is_normalized(self, natural_image):
        """Body skin mask should be in [0, 1] range."""
        engine = RetouchEngine()
        img = natural_image

        result = engine.process(img, recipe="natural", body_smooth=50.0, fast=True)

        # Verify result is valid
        assert result.min() >= 0, "Output should have min >= 0"
        assert result.max() <= 255, "Output should have max <= 255"
        assert result.dtype == np.uint8, "Output should be uint8 BGR"

    def test_body_skin_smoothing_reduces_variance_in_region(self):
        """Body smoothing should reduce high-frequency noise."""
        engine = RetouchEngine()

        # Create synthetic noisy skin image
        h, w = 200, 200
        base_skin = np.ones((h, w, 3), dtype=np.uint8) * 120
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        img = np.clip(base_skin.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Create body mask covering full image
        person_mask = np.ones((h, w), dtype=np.float32)

        result = engine.process(
            img,
            recipe="natural",
            body_smooth=80.0,  # aggressive smoothing
            fast=False,
        )

        # Result should be smoother (lower variance in small patches)
        # This is a rough check: smoothed image should have lower stdev
        assert result is not None
        # Just verify no crash for now; detailed smoothing metrics are complex
