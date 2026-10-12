"""Tests for shine/oil removal (SkinProcessor.shine_removal).

Validates detection, compression, chroma reconstruction, eye exclusion,
over-removal guard, and non-skin exclusion.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from retouch.skin import SkinProcessor


class TestShineRemoval:
    """Suite for shine_removal method."""

    @pytest.fixture
    def processor(self) -> SkinProcessor:
        """Instantiate SkinProcessor."""
        return SkinProcessor()

    def _construct_synthetic_shine_face(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a synthetic face-like image with a bright, desaturated shine patch.

        Uses realistic BGR values (not LAB values chosen to accidentally sit near
        cv2's a/b=128 offset, which previously masked a chroma-computation bug):
        skin_bgr=(140, 170, 200) is a warm skin tone, shine_bgr=(235, 240, 245) is
        a near-white specular highlight. In cv2's raw (uncentered) LAB convention
        these come out to roughly skin=(183, 134, 148), shine=(243, 129, 131) —
        centered chroma (subtracting the 128 offset) is ~20.9 for skin and ~3.2
        for the shine patch, a real and detectable difference.

        Returns:
            (img_bgr, skin_mask, eyes_mask): uint8 BGR image, float32 skin mask,
            and optional eyes mask for testing exclusion.
        """
        h, w = 256, 256
        skin_bgr = (140, 170, 200)
        shine_bgr = (235, 240, 245)

        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        img_bgr[:, :] = skin_bgr
        img_bgr[100:150, 100:150] = shine_bgr

        # Skin mask: include main region, exclude border
        skin_mask = np.zeros((h, w), dtype=np.float32)
        skin_mask[50:200, 50:200] = 1.0

        # Eyes mask: overlaps part of the shine patch (100:150,100:150), simulating
        # a catchlight sitting inside a bright/low-chroma region — the scenario the
        # eye-exclusion guard must actually protect against.
        eyes_mask = np.zeros((h, w), dtype=np.float32)
        eyes_mask[110:130, 110:130] = 1.0

        return img_bgr, skin_mask, eyes_mask

    def test_early_return_zero_strength(self, processor: SkinProcessor) -> None:
        """strength=0 → unchanged output."""
        img, skin_mask, _ = self._construct_synthetic_shine_face()
        result = processor.shine_removal(img, skin_mask, strength=0)
        np.testing.assert_array_equal(result, img)

    def test_early_return_no_skin_mask(self, processor: SkinProcessor) -> None:
        """skin_mask=None → unchanged output."""
        img, _, _ = self._construct_synthetic_shine_face()
        result = processor.shine_removal(img, None, strength=80)
        np.testing.assert_array_equal(result, img)

    def test_shine_detection_and_compression(self, processor: SkinProcessor) -> None:
        """Shine patch L should decrease meaningfully at strength=80.

        Uses realistic BGR values (skin=(140,170,200), shine=(235,240,245)) which
        translate to cv2 raw LAB skin≈(183,134,148), shine≈(243,129,131). Verified
        by direct measurement (2026-07-03, post chroma-offset fix) that strength=80
        produces L: 243.00 -> 234.54 (reduction of 8.46 levels). This asserts a
        real, non-trivial, monotonic-with-strength reduction — not a no-crash check.
        """
        img, skin_mask, _ = self._construct_synthetic_shine_face()

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        shine_region = lab_orig[100:150, 100:150, 0]
        L_shine_before = float(np.mean(shine_region))

        non_shine_region = lab_orig[50:100, 50:100, 0]
        L_skin = float(np.mean(non_shine_region))

        # Sanity: baseline fixture values match the documented cv2 conversion
        assert abs(L_shine_before - 243.0) < 1.0, f"Fixture drifted: L_shine_before={L_shine_before}"
        assert abs(L_skin - 183.0) < 1.0, f"Fixture drifted: L_skin={L_skin}"

        result = processor.shine_removal(img, skin_mask, strength=80)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_shine_after = float(np.mean(lab_result[100:150, 100:150, 0]))

        L_reduction = L_shine_before - L_shine_after
        # Real threshold: at least 5 L-levels of reduction (measured actual: 8.46)
        assert L_reduction >= 5.0, (
            f"Shine L reduced only {L_reduction:.2f} levels. "
            f"Before={L_shine_before:.2f}, After={L_shine_after:.2f}, Skin={L_skin:.2f}. "
            f"Expected >=5.0 (measured baseline ~8.46 at strength=80)."
        )
        # Must move toward skin, not away from it or unchanged
        assert L_shine_after < L_shine_before, "Shine L did not decrease at all"

    def test_compression_scales_monotonically_with_strength(self, processor: SkinProcessor) -> None:
        """L reduction should increase monotonically as strength increases.

        Verified numerically (post-fix): strength 20/50/80/100 give L_after of
        approximately 240.30 / 237.51 / 234.54 / 232.84 (starting from L=243.00).
        """
        img, skin_mask, _ = self._construct_synthetic_shine_face()
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_before = float(np.mean(lab_orig[100:150, 100:150, 0]))

        L_afters = []
        for strength in (20, 50, 80, 100):
            result = processor.shine_removal(img, skin_mask, strength=strength)
            lab_r = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            L_afters.append(float(np.mean(lab_r[100:150, 100:150, 0])))

        # Strictly decreasing L as strength increases (more removal = lower L)
        for i in range(len(L_afters) - 1):
            assert L_afters[i] > L_afters[i + 1], (
                f"L did not decrease monotonically with strength: {L_afters}"
            )
        # All values below the original
        assert all(v < L_before for v in L_afters), f"Some values not below baseline {L_before}: {L_afters}"

    def test_chroma_reconstruction(self, processor: SkinProcessor) -> None:
        """Shine patch chroma AND a/b values should move toward actual surrounding skin.

        Verified numerically (post chroma-offset fix): raw cv2 LAB a/b start at
        shine=(129.00, 131.00) and surrounding skin sits at (134.00, 148.00).
        After strength=80 processing, measured a/b move to (129.51, 132.72) —
        real movement toward the skin's actual a/b, not toward the neutral
        128/128 point (which was the failure mode of the old chroma formula:
        it couldn't discriminate real skin color from noise, so any accidental
        "improvement" wasn't provably directional). Centered chroma moves from
        3.16 -> 4.95, trending toward skin's 20.88 — real magnitude, real direction.
        """
        img, skin_mask, _ = self._construct_synthetic_shine_face()

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        shine_a_before = float(np.mean(lab_orig[100:150, 100:150, 1]))
        shine_b_before = float(np.mean(lab_orig[100:150, 100:150, 2]))
        chroma_before = float(np.sqrt((shine_a_before - 128.0) ** 2 + (shine_b_before - 128.0) ** 2))

        skin_a = float(np.mean(lab_orig[0:50, 0:50, 1]))
        skin_b = float(np.mean(lab_orig[0:50, 0:50, 2]))
        chroma_skin = float(np.sqrt((skin_a - 128.0) ** 2 + (skin_b - 128.0) ** 2))

        # Sanity: fixture matches documented cv2 conversion
        assert abs(shine_a_before - 129.0) < 1.0 and abs(shine_b_before - 131.0) < 1.0
        assert abs(skin_a - 134.0) < 1.0 and abs(skin_b - 148.0) < 1.0

        result = processor.shine_removal(img, skin_mask, strength=80)

        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        shine_a_after = float(np.mean(lab_result[100:150, 100:150, 1]))
        shine_b_after = float(np.mean(lab_result[100:150, 100:150, 2]))
        chroma_after = float(np.sqrt((shine_a_after - 128.0) ** 2 + (shine_b_after - 128.0) ** 2))

        # a and b must each move toward the ACTUAL skin a/b (134, 148), not toward 128
        a_delta = shine_a_after - shine_a_before
        b_delta = shine_b_after - shine_b_before
        assert a_delta > 0.1, (
            f"a channel did not move toward skin a={skin_a:.1f}: "
            f"before={shine_a_before:.2f}, after={shine_a_after:.2f}"
        )
        assert b_delta > 0.5, (
            f"b channel did not move toward skin b={skin_b:.1f}: "
            f"before={shine_b_before:.2f}, after={shine_b_after:.2f}"
        )
        # Both must have moved strictly closer to the real skin values (not overshooting)
        assert abs(shine_a_after - skin_a) < abs(shine_a_before - skin_a), "a moved away from skin"
        assert abs(shine_b_after - skin_b) < abs(shine_b_before - skin_b), "b moved away from skin"

        # Chroma increased measurably (real magnitude, matching the verified ~1.8 gain)
        chroma_gain = chroma_after - chroma_before
        assert chroma_gain > 1.0, (
            f"Chroma reconstruction too weak: before={chroma_before:.2f}, "
            f"after={chroma_after:.2f}, skin={chroma_skin:.2f}, gain={chroma_gain:.2f}. "
            f"Expected gain > 1.0 (measured baseline ~1.79)."
        )

    def test_eye_exclusion(self, processor: SkinProcessor) -> None:
        """Eyes mask should protect a bright low-chroma region (catchlight) from modification.

        The fixture's eyes_mask (110:130,110:130) sits inside the shine patch
        (100:150,100:150) — the realistic catchlight-inside-shine scenario. Verified
        numerically: with eyes_mask=None, that sub-region gets compressed along with
        the rest of the shine patch (L drops meaningfully); with eyes_mask supplied,
        it stays within ~0.1 L of the untouched original (243.00).
        """
        img, skin_mask, eyes_mask = self._construct_synthetic_shine_face()

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_eye_region_orig = float(np.mean(lab_orig[110:130, 110:130, 0]))

        result_no_eye_mask = processor.shine_removal(img, skin_mask, strength=80, eyes_mask=None)
        result_with_eye_mask = processor.shine_removal(img, skin_mask, strength=80, eyes_mask=eyes_mask)

        lab_no_mask = cv2.cvtColor(result_no_eye_mask, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_with_mask = cv2.cvtColor(result_with_eye_mask, cv2.COLOR_BGR2LAB).astype(np.float32)

        eye_region_no_mask = float(np.mean(lab_no_mask[110:130, 110:130, 0]))
        eye_region_with_mask = float(np.mean(lab_with_mask[110:130, 110:130, 0]))

        # Without eye protection, the region (being inside the shine patch) IS
        # compressed away from its original bright value.
        assert L_eye_region_orig - eye_region_no_mask > 3.0, (
            f"Sanity check failed: unprotected eye-overlapping region should be "
            f"compressed by shine removal. orig={L_eye_region_orig:.2f}, "
            f"no_mask_result={eye_region_no_mask:.2f}"
        )

        # With eye protection, the region must stay close to its original value.
        protected_change = abs(eye_region_with_mask - L_eye_region_orig)
        assert protected_change < 2.0, (
            f"Eye exclusion failed: protected region changed by {protected_change:.2f} L-levels "
            f"(orig={L_eye_region_orig:.2f}, with_mask_result={eye_region_with_mask:.2f}). "
            f"Expected < 2.0."
        )

        # And the protected vs. unprotected results must differ substantially,
        # proving the mask is actually doing something (not a no-op).
        L_diff = abs(eye_region_with_mask - eye_region_no_mask)
        assert L_diff > 2.0, (
            f"Eye mask had no protective effect: with_mask={eye_region_with_mask:.2f}, "
            f"no_mask={eye_region_no_mask:.2f}, diff={L_diff:.2f}. Expected > 2.0."
        )

    def test_over_removal_guard(self, processor: SkinProcessor) -> None:
        """At strength=100, shine patch should retain >=25% of its prominence above local median.

        Measures the ratio: (final_L - local_median_L) / (original_L - local_median_L).
        Verified numerically (post-fix): orig=243.00, final=232.84, local_median=183.00,
        giving residual_ratio ~0.831 — comfortably above the 0.25 floor.
        """
        img, skin_mask, _ = self._construct_synthetic_shine_face()

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        shine_region_orig = lab_orig[100:150, 100:150, 0]
        L_shine_orig = float(np.mean(shine_region_orig))

        # Local median: surrounding skin L
        local_median = float(np.mean(lab_orig[50:100, 50:100, 0]))

        # Process at max strength
        result = processor.shine_removal(img, skin_mask, strength=100)

        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        shine_region_final = lab_result[100:150, 100:150, 0]
        L_shine_final = float(np.mean(shine_region_final))

        # Calculate residual prominence ratio
        original_prominence = L_shine_orig - local_median
        final_prominence = L_shine_final - local_median
        residual_ratio = final_prominence / (original_prominence + 1e-6)

        # Expect at least 25% residual prominence to show over-removal guard is active
        assert residual_ratio >= 0.25, (
            f"Over-removal guard inactive: residual_ratio={residual_ratio:.2f}. "
            f"Original prominence={original_prominence:.1f}, "
            f"final prominence={final_prominence:.1f}. "
            f"Expected ≥0.25."
        )

    def test_non_skin_exclusion(self, processor: SkinProcessor) -> None:
        """A bright low-chroma pixel OUTSIDE skin_mask must be completely unaffected.

        Creates a bright spot outside the skin mask and verifies it is unchanged.
        """
        h, w = 256, 256
        img = np.ones((h, w, 3), dtype=np.uint8) * 100  # Neutral gray

        # Skin mask: just a small region
        skin_mask = np.zeros((h, w), dtype=np.float32)
        skin_mask[100:150, 100:150] = 1.0

        # Create a bright low-chroma region OUTSIDE skin_mask (upper left)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[30:60, 30:60, 0] = 230.0  # Bright
        lab[30:60, 30:60, 1] = 128.0  # Neutral a
        lab[30:60, 30:60, 2] = 128.0  # Neutral b
        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Process
        result = processor.shine_removal(img, skin_mask, strength=100)

        # Verify outside region is unchanged
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

        outside_L_orig = lab_orig[30:60, 30:60, 0]
        outside_L_result = lab_result[30:60, 30:60, 0]

        L_change = float(np.mean(np.abs(outside_L_result - outside_L_orig)))
        assert L_change < 1.0, (
            f"Non-skin region was modified: L changed by {L_change:.1f}. "
            f"Expected <1.0 (essentially unchanged)."
        )

    def test_integration_end_to_end(self, processor: SkinProcessor) -> None:
        """End-to-end: typical portrait with strength=50 should complete without error.

        Verifies that processing completes without error and produces a valid image.
        """
        img, skin_mask, eyes_mask = self._construct_synthetic_shine_face()

        # Process at moderate strength
        result = processor.shine_removal(img, skin_mask, strength=50, eyes_mask=eyes_mask)

        # Verify output is valid uint8 BGR
        assert result.dtype == np.uint8
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 255
        assert not np.isnan(result).any(), "Output contains NaN values"
        assert not np.isinf(result).any(), "Output contains Inf values"


class TestShineRemovalEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.fixture
    def processor(self) -> SkinProcessor:
        """Instantiate SkinProcessor."""
        return SkinProcessor()

    def test_empty_skin_mask(self, processor: SkinProcessor) -> None:
        """All-zero skin_mask → unchanged output."""
        h, w = 256, 256
        img = np.ones((h, w, 3), dtype=np.uint8) * 180
        skin_mask = np.zeros((h, w), dtype=np.float32)

        result = processor.shine_removal(img, skin_mask, strength=80)
        np.testing.assert_array_equal(result, img)

    def test_small_shine_region(self, processor: SkinProcessor) -> None:
        """Very small shine region (1x1 pixel) should be handled gracefully."""
        h, w = 256, 256
        img = np.ones((h, w, 3), dtype=np.uint8) * 180

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[128, 128, :] = [230.0, 128.0, 128.0]
        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        skin_mask = np.ones((h, w), dtype=np.float32)

        # Should complete without error
        result = processor.shine_removal(img, skin_mask, strength=80)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_full_image_shine(self, processor: SkinProcessor) -> None:
        """Entire image is bright and desaturated (extreme shine).

        Verify processing doesn't produce NaN or clipping artifacts.
        """
        h, w = 256, 256
        lab = np.ones((h, w, 3), dtype=np.float32)
        lab[:, :, 0] = 240.0  # Very bright
        lab[:, :, 1] = 128.0  # Neutral a
        lab[:, :, 2] = 128.0  # Neutral b

        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        skin_mask = np.ones((h, w), dtype=np.float32)

        result = processor.shine_removal(img, skin_mask, strength=100)

        # Verify no NaN, Inf, or out-of-range values
        assert np.isfinite(result.astype(np.float32)).all()
        assert result.min() >= 0 and result.max() <= 255
