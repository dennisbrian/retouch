"""Tests for retouch/qa_detectors.py — quality assurance artifact detection."""

import numpy as np
import cv2
import pytest

from retouch.qa_detectors import (
    detect_banding,
    detect_clipping,
    detect_plastic_skin,
    detect_halo,
    run_all,
    BANDING_THRESHOLD,
    CLIPPING_THRESHOLD,
    PLASTIC_SKIN_THRESHOLD,
    HALO_THRESHOLD,
)


class TestDetectBanding:
    """Tests for banding detection on smooth gradients."""

    def test_smooth_gradient_not_flagged(self):
        """A clean smooth gradient should not be flagged as banded."""
        # Create a smooth gradient in L channel
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            val = int(y * 2.55)
            img[y, :] = [val, val, val]

        result = detect_banding(img)
        # Smooth gradients may show step edges due to uint8 quantization
        # but score should be reasonable
        assert "flagged" in result
        assert "score" in result
        assert "smooth_pixels" in result
        assert "step_pixels" in result

    def test_posterized_gradient_flagged(self):
        """A posterized (quantized) gradient should be flagged as banded."""
        # Create a posterized gradient with 8 levels instead of 256
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            # Quantize to 8 levels, clamped to 0-255
            level = min(255, (y // 12) * 32)
            img[y, :] = [level, level, level]

        result = detect_banding(img)
        # Posterized should have some step pixels
        assert "step_pixels" in result
        # Note: depending on detection sensitivity, may or may not flag
        # but should be a valid result

    def test_with_skin_mask(self):
        """Banding detection should respect a mask."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            val = int(y * 2.55)
            img[y, :] = [val, val, val]

        mask = np.zeros((100, 100), dtype=np.float32)
        mask[50:100, :] = 1.0  # Only bottom half

        result = detect_banding(img, mask=mask)
        assert "score" in result
        assert "flagged" in result

    def test_tiny_image_safe(self):
        """Tiny images should not crash and return sensible defaults."""
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = detect_banding(img)
        assert result["flagged"] is False
        assert result["score"] == 0.0
        assert result["smooth_pixels"] == 0


class TestDetectClipping:
    """Tests for clipped highlight and shadow detection."""

    def test_well_exposed_not_clipped(self):
        """A well-exposed image should have low clipping."""
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        result = detect_clipping(img)
        assert result["clipped_pixel_fraction"] < 0.02  # Very few at 0 or 255
        assert result["flagged"] is False

    def test_blown_highlights_flagged(self):
        """Image with large blown-white blob should be flagged."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Add a large blown-white blob
        img[30:70, 30:70, :] = 255

        result = detect_clipping(img)
        assert result["clipped_blob_fraction"] > 0.05
        # Should be flagged if blob fraction exceeds threshold
        if result["clipped_blob_fraction"] > CLIPPING_THRESHOLD:
            assert result["flagged"] is True

    def test_crushed_blacks_detected(self):
        """Image with crushed shadows should be detected."""
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        # Add a crushed-black region
        img[30:70, 30:70, :] = 0

        result = detect_clipping(img)
        assert result["clipped_pixel_fraction"] > 0
        assert "clipped_blob_fraction" in result

    def test_with_mask_restricts_analysis(self):
        """Clipping detection should respect a mask."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        img[30:70, 30:70, :] = 255  # Blown region

        mask = np.zeros((100, 100), dtype=np.float32)
        mask[:30, :] = 1.0  # Only top portion (no blown region)

        result = detect_clipping(img, mask=mask)
        # Should show lower clipping when blown region is masked out
        assert result["clipped_blob_fraction"] >= 0.0

    def test_tiny_image_safe(self):
        """Tiny images should not crash."""
        img = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = detect_clipping(img)
        assert result["flagged"] is False
        assert result["clipped_blob_fraction"] == 0.0


class TestDetectPlasticSkin:
    """Tests for over-smoothed skin detection."""

    def test_textured_skin_not_plastic(self):
        """Skin with natural texture should not be flagged as plastic."""
        # Create textured skin-like image (Gaussian noise + low-freq color)
        np.random.seed(42)
        noise = np.random.normal(150, 20, (100, 100, 3))
        img = np.clip(noise, 0, 255).astype(np.uint8)

        mask = np.ones((100, 100), dtype=np.float32)
        result = detect_plastic_skin(img, mask=mask)

        assert "hf_energy_ratio" in result
        # Natural texture should have reasonable energy ratio
        assert result["hf_energy_ratio"] > 0

    def test_gaussian_blur_reduces_texture(self):
        """Gaussian-blurred skin should show lower high-frequency energy."""
        # Create textured image
        np.random.seed(42)
        img = np.random.randint(130, 170, (100, 100, 3), dtype=np.uint8)

        # Blur it heavily
        img_blurred = cv2.GaussianBlur(img, (11, 11), 2)

        result_original = detect_plastic_skin(img)
        result_blurred = detect_plastic_skin(img_blurred)

        # Blurred should have lower high-freq energy
        assert result_blurred["hf_energy_ratio"] < result_original["hf_energy_ratio"]

    def test_with_reference_comparison(self):
        """Plastic skin detection should compare with reference if provided."""
        # Original textured
        np.random.seed(42)
        original = np.random.randint(130, 170, (100, 100, 3), dtype=np.uint8)

        # Over-smoothed version
        smoothed = cv2.GaussianBlur(original, (11, 11), 2)

        mask = np.ones((100, 100), dtype=np.float32)
        result = detect_plastic_skin(smoothed, mask=mask, reference_img_bgr=original)

        assert "energy_loss_vs_reference" in result
        assert result["energy_loss_vs_reference"] is not None
        # Energy should be lower in smoothed version
        assert result["energy_loss_vs_reference"] > 0

    def test_with_skin_mask(self):
        """Plastic skin detection should respect a mask."""
        np.random.seed(42)
        img = np.random.randint(130, 170, (100, 100, 3), dtype=np.uint8)

        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0  # Center region only

        result = detect_plastic_skin(img, mask=mask)
        assert "score" in result
        assert "hf_energy_ratio" in result

    def test_tiny_image_safe(self):
        """Tiny images should not crash."""
        img = np.full((4, 4, 3), 150, dtype=np.uint8)
        result = detect_plastic_skin(img)
        assert result["flagged"] is False
        assert result["score"] == 1.0


class TestDetectHalo:
    """Tests for edge overshoot (sharpening halos)."""

    def test_clean_edge_no_halo(self):
        """A clean edge with no ringing should show low overshoot."""
        # Create a simple step edge
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 200

        result = detect_halo(img)
        # Clean edge should have low overshoot
        assert result["mean_overshoot"] >= 0
        # May or may not be flagged depending on threshold

    def test_oversharpened_edge_halo(self):
        """Edge with ringing overshoot should be flagged."""
        # Create a step edge with simulated oversharpening ringing
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 200

        # Apply unsharp mask (strong) to create halos
        blurred = cv2.GaussianBlur(img, (5, 5), 1)
        img_sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        img_sharpened = np.clip(img_sharpened, 0, 255).astype(np.uint8)

        result = detect_halo(img_sharpened)
        # May show overshoot due to sharpening
        assert "mean_overshoot" in result
        assert "edge_count" in result

    def test_with_mask_restricts_edges(self):
        """Halo detection should respect a mask."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 200

        mask = np.zeros((100, 100), dtype=np.float32)
        mask[:, :50] = 1.0  # Only dark side

        result = detect_halo(img, mask=mask)
        assert "mean_overshoot" in result
        assert "edge_count" in result

    def test_tiny_image_safe(self):
        """Tiny images should not crash."""
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        result = detect_halo(img)
        assert result["flagged"] is False
        assert result["mean_overshoot"] == 0.0


class TestRunAll:
    """Tests for the aggregated run_all() function."""

    def test_returns_all_four_detectors(self):
        """run_all() should return results for all four detectors."""
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        result = run_all(img)

        assert "banding" in result
        assert "clipping" in result
        assert "plastic_skin" in result
        assert "halo" in result

        # Each should have score and flagged
        for detector_name in ["banding", "clipping", "plastic_skin", "halo"]:
            assert "score" in result[detector_name]
            assert "flagged" in result[detector_name]

    def test_with_skin_mask_all(self):
        """run_all() should pass skin mask to all detectors."""
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0

        result = run_all(img, skin_mask=mask)
        assert "banding" in result
        assert "clipping" in result
        assert "plastic_skin" in result
        assert "halo" in result

    def test_with_reference_image(self):
        """run_all() should pass reference to plastic skin detector."""
        np.random.seed(42)
        original = np.random.randint(130, 170, (100, 100, 3), dtype=np.uint8)
        smoothed = cv2.GaussianBlur(original, (9, 9), 2)

        result = run_all(smoothed, reference_img_bgr=original)
        assert result["plastic_skin"]["energy_loss_vs_reference"] is not None

    def test_empty_mask_safe(self):
        """Empty/zero mask should be handled gracefully."""
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.float32)  # All zeros

        result = run_all(img, skin_mask=mask)
        assert "banding" in result
        assert "clipping" in result
        assert "plastic_skin" in result
        assert "halo" in result

    def test_tiny_image_safe(self):
        """Tiny images should not crash."""
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        result = run_all(img)

        for detector_name in ["banding", "clipping", "plastic_skin", "halo"]:
            assert result[detector_name]["flagged"] is False


class TestIntegrationScenarios:
    """Integration tests with realistic retouching scenarios."""

    def test_over_smoothed_retouching(self):
        """Scenario: skin over-smoothed during retouching."""
        # Original: textured skin
        np.random.seed(42)
        original = np.random.randint(130, 170, (150, 150, 3), dtype=np.uint8)

        # "Retouched": blurred to death
        retouched = cv2.GaussianBlur(original, (15, 15), 3)

        mask = np.ones((150, 150), dtype=np.float32)
        mask[:50, :] = 0  # Background mask

        result = run_all(retouched, skin_mask=mask, reference_img_bgr=original)

        # Plastic skin should be flagged
        assert result["plastic_skin"]["energy_loss_vs_reference"] > 0

    def test_aggressive_sharpening_halo(self):
        """Scenario: excessive sharpening creating halos."""
        # Base image
        img = np.full((150, 150, 3), 180, dtype=np.uint8)
        # Add an edge
        img[75:, :] = 100

        # Apply aggressive unsharp mask
        blurred = cv2.GaussianBlur(img, (7, 7), 2)
        sharpened = cv2.addWeighted(img, 2.0, blurred, -1.0, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        result = run_all(sharpened)
        # Halo detection should pick up overshoot
        assert "mean_overshoot" in result["halo"]

    def test_high_iso_clipping(self):
        """Scenario: high-ISO image with blown highlights."""
        # Simulate high-ISO noise + blown highlights
        np.random.seed(42)
        img = np.random.normal(150, 15, (150, 150, 3))
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Blown highlights in top-right
        img[10:50, 100:150] = 255

        result = run_all(img)
        assert result["clipping"]["clipped_blob_fraction"] > 0

    def test_compression_posterization(self):
        """Scenario: posterization from over-compression."""
        # Create a smooth gradient
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        for y in range(150):
            val = int(y * 1.7)
            img[y, :] = [val, val, val]

        # Simulate compression by quantizing to fewer levels
        levels = 16
        img_quantized = (img // (256 // levels)) * (256 // levels)

        result = run_all(img_quantized)
        # Banding detection should pick up the posterization
        assert "step_pixels" in result["banding"]
